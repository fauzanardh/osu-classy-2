import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from xformers.ops import memory_efficient_attention


def FeedForward(in_dim, mult):
    return nn.Sequential(
        nn.GroupNorm(1, in_dim),
        nn.Conv1d(in_dim, in_dim * mult, 1),
        nn.GELU(),
        nn.GroupNorm(1, in_dim * mult),
        nn.Conv1d(in_dim * mult, in_dim, 1),
    )


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
    def __init__(self, in_dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        h_dim = dim_head * heads
        self.dim_head = dim_head

        self.to_qkv = nn.Conv1d(in_dim, h_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(h_dim, in_dim, 1)

    def attn(self, q, k, v):
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        # b h n d -> b (h d) n
        out = rearrange(out, "b h n d -> b (h d) n")
        return out

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        out = self.attn(*(t.unflatten(1, (self.heads, -1)) for t in qkv))
        out = self.to_out(out)
        return out


class LinearAttention(Attention):
    def attn(self, q, k, v):
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        ctx = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", ctx, q)
        # b h c l -> b (h c) l
        # out = out.view(out.shape[0], -1, out.shape[-1])
        # use rearrange from einops
        out = rearrange(out, "b h c l -> b (h c) l")
        return out


class FlashAttention(Attention):
    def attn(self, q, k, v):
        out_dtype = q.dtype
        # b h d n -> b n h d
        q = rearrange(q, "b h d n -> b n h d").contiguous()
        k = rearrange(k, "b h d n -> b n h d").contiguous()
        v = rearrange(v, "b h d n -> b n h d").contiguous()

        # to fp16
        q = q.half()
        k = k.half()
        v = v.half()
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        # b n h d -> b (h d) n
        out = rearrange(out, "b n h d -> b (h d) n")
        return out.to(out_dtype)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        depth=2,
        heads=8,
        dim_head=32,
        ff_mult=2,
        use_flash_attn=False,
        use_linear_attn=False,
    ):
        super().__init__()
        assert not (use_flash_attn and use_linear_attn), "can't use both"

        if use_flash_attn:
            attn_block = FlashAttention
        elif use_linear_attn:
            attn_block = LinearAttention
        else:
            attn_block = Attention

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        attn_block(in_dim, heads, dim_head),
                        FeedForward(in_dim, ff_mult),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim) if norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(in_dim, out_dim, 7, padding=3),
            nn.GroupNorm(1, out_dim) if norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(out_dim, out_dim, 7, padding=3),
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
        num_res_blocks=2,
        attn_heads=8,
        attn_dim_head=32,
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
                                res_block(dim_in if i == 0 else dim_out, dim_out)
                                for i in range(num_res_blocks)
                            ]
                        ),
                        TransformerBlock(
                            dim_out,
                            use_flash_attn=use_flash_attn,
                            use_linear_attn=use_linear_attn,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
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
        self.mid_attn = TransformerBlock(
            mid_dim,
            use_flash_attn=use_flash_attn,
            use_linear_attn=use_linear_attn,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )
        self.mid_block2 = res_block(mid_dim, mid_dim)

        # end
        self.norm = nn.GroupNorm(1, mid_dim)
        self.conv_out = nn.Conv1d(mid_dim, z_dim, 7, padding=3)

    def forward(self, x):
        # downsample
        x = self.init_conv(x)

        for blocks, attn, downsample in self.downs:
            for block in blocks:
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
        num_res_blocks=2,
        attn_heads=8,
        attn_dim_head=32,
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
        self.mid_attn = TransformerBlock(
            mid_dim,
            use_flash_attn=use_flash_attn,
            use_linear_attn=use_linear_attn,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )
        self.mid_block2 = res_block(mid_dim, mid_dim)

        # up
        self.up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                res_block(dim_in if i == 0 else dim_out, dim_out)
                                for i in range(num_res_blocks)
                            ]
                        ),
                        TransformerBlock(
                            dim_out,
                            use_flash_attn=use_flash_attn,
                            use_linear_attn=use_linear_attn,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
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
        for blocks, attn, upsample in self.up:
            for block in blocks:
                x = block(x)
            x = attn(x)
            x = upsample(x)

        # end
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class VQEmbedding(nn.Module):
    def __init__(
        self,
        n_emb,
        emb_dim,
        beta=0.25,
    ):
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.emb = nn.Embedding(n_emb, emb_dim)
        self.emb.weight.data.uniform_(-1 / n_emb, 1 / n_emb)

    def forward(self, z):
        z = rearrange(z, "b c l -> b l c")
        z_flat = z.reshape(-1, self.emb_dim)

        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.emb.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flat, self.emb.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=-1)
        z_q = self.emb(encoding_indices).view_as(z)
        perplexity = None
        encodings = None

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, "b l c -> b c l")

        return z_q, loss, (perplexity, encodings, encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.emb(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = rearrange(z_q, "b l c -> b c l")

        return z_q


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
        attn_heads=8,
        attn_dim_head=32,
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
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )

        self.vq = VQEmbedding(n_emb, emb_dim)
        self.quant_conv = nn.Conv1d(z_dim, emb_dim, 1)
        self.post_quant_conv = nn.Conv1d(emb_dim, z_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.vq(h)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        return self.quant_conv(h)

    def decode(self, quantized):
        quantized = self.post_quant_conv(quantized)
        return self.decoder(quantized)

    def forward(self, x, return_indices=False):
        quantized, loss, (_, _, ind) = self.encode(x)
        decoded = self.decode(quantized)
        decoded = torch.tanh(decoded)
        if return_indices:
            return decoded, loss, ind
        else:
            return decoded, loss
