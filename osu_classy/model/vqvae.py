import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from xformers.ops import memory_efficient_attention
from vector_quantize_pytorch import VectorQuantize


def l2norm(t):
    return F.normalize(t, dim=-1)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return gate * F.silu(x)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.GroupNorm(1, dim),
        nn.Conv1d(dim, inner_dim * 2, 1, bias=False),
        SwiGLU(),
        nn.GroupNorm(1, inner_dim),
        nn.Conv1d(inner_dim, dim, 1, bias=False),
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, *kwargs) + x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels, in_channels, 4, stride=2, padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, in_channels, 4, stride=2, padding=1, padding_mode="reflect"
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_dim, heads=16, dim_head=64, scale=8):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        h_dim = dim_head * heads
        self.dim_head = dim_head
        self.to_qkv = nn.Conv1d(in_dim, h_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(h_dim, in_dim, 1)

    def attn(self, q, k, v):
        q = rearrange(q, "b h d n -> b n h d")
        k = rearrange(k, "b h d n -> b n h d")
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        q = rearrange(q, "b n h d -> b h d n")
        k = rearrange(k, "b n h d -> b h d n")

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b (h d) n")
        return out

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        out = self.attn(*(t.unflatten(1, (self.heads, -1)) for t in qkv))
        out = self.to_out(out)
        return out


class LinearAttention(Attention):
    def attn(self, q, k, v):
        q = rearrange(q, "b h d n -> b n h d")
        k = rearrange(k, "b h d n -> b n h d")
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        q = rearrange(q, "b n h d -> b h d n")
        k = rearrange(k, "b n h d -> b h d n")

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        ctx = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", ctx, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        return out


class FlashAttention(Attention):
    def attn(self, q, k, v):
        out_dtype = q.dtype
        q = rearrange(q, "b h d n -> b n h d").contiguous()
        k = rearrange(k, "b h d n -> b n h d").contiguous()
        v = rearrange(v, "b h d n -> b n h d").contiguous()

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # to fp16
        q = q.half()
        k = k.half()
        v = v.half()
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, "b n h d -> b (h d) n")
        return out.to(out_dtype)


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        in_dim,
        attn_heads=16,
        attn_dim_head=64,
        use_flash_attn=False,
        use_linear_attn=False,
        ff_mult=4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.ff_mult = ff_mult

        if use_flash_attn:
            attn_block = FlashAttention
        elif use_linear_attn:
            attn_block = LinearAttention
        else:
            attn_block = Attention

        self.attn = attn_block(in_dim, attn_heads, attn_dim_head)
        self.ff = FeedForward(in_dim, ff_mult)
        self.norm = nn.GroupNorm(1, in_dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        x = self.norm(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim) if norm else nn.Identity(),
            nn.Conv1d(in_dim, out_dim, 7, padding=3),
            nn.SiLU(),
            nn.GroupNorm(1, out_dim) if norm else nn.Identity(),
            nn.Conv1d(out_dim, out_dim, 7, padding=3),
            nn.SiLU(),
        )

        if self.in_dim != self.out_dim:
            self.shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        h = x
        h = self.net(h)

        if self.in_dim != self.out_dim:
            x = self.shortcut(x)

        return x + h


class ConvNextBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mult=2):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ds_conv = nn.Conv1d(
            in_dim, in_dim, 7, padding=3, groups=in_dim, padding_mode="reflect"
        )
        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim),
            nn.Conv1d(in_dim, out_dim * mult, 7, padding=3, padding_mode="reflect"),
            nn.SiLU(),
            nn.GroupNorm(1, out_dim * mult),
            nn.Conv1d(out_dim * mult, out_dim, 7, padding=3, padding_mode="reflect"),
        )
        if self.in_dim != self.out_dim:
            self.shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        h = self.ds_conv(x)
        h = self.net(h)

        if self.in_dim != self.out_dim:
            x = self.shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        z_dim,
        dim_mult=(1, 2, 4, 8),
        use_flash_attn=False,
        use_linear_attn=False,
        use_conv_next=False,
        num_res_blocks=3,
        attn_depth=2,
        attn_heads=16,
        attn_dim_head=64,
        ff_mult=4,
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(in_dim, h_dim, 7, padding=3)

        assert not (use_flash_attn and use_linear_attn), "can't use both attn types"

        res_block = ConvNextBlock if use_conv_next else ResnetBlock

        h_dims = [h_dim * d for d in dim_mult]
        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)

        # down
        self.downs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                res_block(
                                    dim_in if i == 0 else dim_out,
                                    dim_out,
                                )
                                for i in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlocks(
                                    dim_out,
                                    attn_heads,
                                    attn_dim_head,
                                    use_flash_attn,
                                    use_linear_attn,
                                    ff_mult,
                                )
                                for _ in range(attn_depth)
                            ]
                        ),
                        Downsample(dim_out)
                        if ind < (num_layers - 1)
                        else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out)
            ]
        )

        # middle
        mid_dim = h_dims[-1]
        self.mid_block1 = res_block(mid_dim, mid_dim)
        self.mid_attn = TransformerBlocks(
            mid_dim,
            attn_heads,
            attn_dim_head,
            use_flash_attn,
            use_linear_attn,
            ff_mult,
        )
        self.mid_block2 = res_block(mid_dim, mid_dim)

        # end
        self.norm = nn.GroupNorm(1, mid_dim)
        self.conv_out = nn.Conv1d(mid_dim, z_dim, 7, padding=3)

    def forward(self, x):
        # downsample
        x = self.init_conv(x)

        for blocks, attns, downsample in self.downs:
            for block, attn in zip(blocks, attns):
                x = block(x)
                x = attn(x)
            x = downsample(x)

        # middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # end
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        z_dim,
        dim_mult=(1, 2, 4, 8),
        use_flash_attn=False,
        use_linear_attn=False,
        use_conv_next=False,
        num_res_blocks=3,
        attn_depth=2,
        attn_heads=16,
        attn_dim_head=64,
        ff_mult=4,
    ):
        super().__init__()

        assert not (use_flash_attn and use_linear_attn), "can't use both attn types"

        dim_mult = tuple(reversed(dim_mult))
        h_dims = [h_dim * d for d in dim_mult]
        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)

        res_block = ConvNextBlock if use_conv_next else ResnetBlock

        # middle
        mid_dim = h_dims[0]
        self.init_conv = nn.Conv1d(z_dim, mid_dim, 7, padding=3)
        self.mid_block1 = res_block(mid_dim, mid_dim)
        self.mid_attn = TransformerBlocks(
            mid_dim,
            attn_heads,
            attn_dim_head,
            use_flash_attn,
            use_linear_attn,
            ff_mult,
        )
        self.mid_block2 = res_block(mid_dim, mid_dim)

        # up
        self.up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                res_block(
                                    dim_in if i == 0 else dim_out,
                                    dim_out,
                                )
                                for i in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlocks(
                                    dim_out,
                                    attn_heads,
                                    attn_dim_head,
                                    use_flash_attn,
                                    use_linear_attn,
                                    ff_mult,
                                )
                                for _ in range(attn_depth)
                            ]
                        ),
                        Upsample(dim_out) if ind < (num_layers - 1) else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out)
            ]
        )

        # end
        self.norm = nn.GroupNorm(1, h_dim)
        self.conv_out = nn.Conv1d(h_dim, in_dim, 7, padding=3)

    def forward(self, x):
        # upsample
        x = self.init_conv(x)

        # middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # upsample
        for blocks, attns, upsample in self.up:
            for block, attn in zip(blocks, attns):
                x = block(x)
                x = attn(x)
            x = upsample(x)

        # end
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return torch.tanh(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        z_dim,
        n_emb,
        emb_dim,
        dim_mult=(1, 2, 4, 8),
        use_flash_attn=False,
        use_linear_attn=False,
        use_conv_next=False,
        num_res_blocks=3,
        attn_depth=2,
        attn_heads=16,
        attn_dim_head=64,
        commitment_weight=1.0,
        use_l1_loss=False,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_dim,
            h_dim,
            z_dim,
            dim_mult=dim_mult,
            use_flash_attn=use_flash_attn,
            use_linear_attn=use_linear_attn,
            use_conv_next=use_conv_next,
            num_res_blocks=num_res_blocks,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )
        self.decoder = Decoder(
            in_dim,
            h_dim,
            z_dim,
            dim_mult=dim_mult,
            use_flash_attn=use_flash_attn,
            use_linear_attn=use_linear_attn,
            use_conv_next=use_conv_next,
            num_res_blocks=num_res_blocks,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )

        self.vq = VectorQuantize(
            dim=emb_dim,
            codebook_size=n_emb,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            channel_last=False,
            use_cosine_sim=True,
        )
        self.quant_conv = (
            nn.Conv1d(z_dim, emb_dim, 1) if emb_dim != z_dim else nn.Identity()
        )
        self.post_quant_conv = (
            nn.Conv1d(emb_dim, z_dim, 1) if emb_dim != z_dim else nn.Identity()
        )

        self.recon_loss_fn = F.l1_loss if use_l1_loss else F.mse_loss

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        fmap = self.encoder(fmap)
        fmap = self.quant_conv(fmap)
        return self.vq(fmap)

    def decode(self, quantized):
        quantized = self.post_quant_conv(quantized)
        return self.decoder(quantized)

    def decode_from_ids(self, ids):
        codes = self.codebook[ids]
        fmap = self.vq.project_out(codes)
        fmap = rearrange(fmap, "b l c -> b c l")
        return self.decode(fmap)

    def forward(self, sig, return_recons=False):
        fmap, _, commit_loss = self.encode(sig)
        fmap = self.decode(fmap)

        recon_loss = self.recon_loss_fn(fmap, sig)
        loss = recon_loss + commit_loss

        if return_recons:
            return loss, fmap
        else:
            return loss
