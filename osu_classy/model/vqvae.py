import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from xformers.ops import memory_efficient_attention
from vector_quantize_pytorch import VectorQuantize


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
        out = rearrange(out, "b h c l -> b (h c) l")
        return out


class FlashAttention(Attention):
    def attn(self, q, k, v):
        out_dtype = q.dtype
        q = rearrange(q, "b h d n -> b n h d").contiguous()
        k = rearrange(k, "b h d n -> b n h d").contiguous()
        v = rearrange(v, "b h d n -> b n h d").contiguous()

        # to fp16
        q = q.half()
        k = k.half()
        v = v.half()
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, "b n h d -> b (h d) n")
        return out.to(out_dtype)


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, norm=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim) if norm else nn.Identity(),
            nn.Conv1d(in_dim, out_dim, 7, padding=3),
            nn.SiLU(),
            nn.Dropout(dropout),
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
    def __init__(self, in_dim, out_dim, dropout=0.5, mult=2):
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
            nn.Dropout(dropout),
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
        attn_heads=8,
        attn_dim_head=32,
        dropout=0.5,
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(in_dim, h_dim, 7, padding=3)

        assert not (use_flash_attn and use_linear_attn), "can't use both attn types"

        res_block = ConvNextBlock if use_conv_next else ResnetBlock
        if use_flash_attn:
            attn_block = FlashAttention
        elif use_linear_attn:
            attn_block = LinearAttention
        else:
            attn_block = Attention

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
                                    dropout=dropout,
                                )
                                for i in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                Residual(
                                    PreNorm(
                                        dim_out,
                                        attn_block(dim_out, attn_heads, attn_dim_head),
                                    )
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
        self.mid_block1 = res_block(mid_dim, mid_dim, dropout=dropout)
        self.mid_attn = Residual(
            PreNorm(mid_dim, attn_block(mid_dim, attn_heads, attn_dim_head))
        )
        self.mid_block2 = res_block(mid_dim, mid_dim, dropout=dropout)

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
        attn_heads=8,
        attn_dim_head=32,
        dropout=0.5,
    ):
        super().__init__()

        assert not (use_flash_attn and use_linear_attn), "can't use both attn types"

        dim_mult = tuple(reversed(dim_mult))
        h_dims = [h_dim * d for d in dim_mult]
        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)

        res_block = ConvNextBlock if use_conv_next else ResnetBlock
        if use_flash_attn:
            attn_block = FlashAttention
        elif use_linear_attn:
            attn_block = LinearAttention
        else:
            attn_block = Attention

        # middle
        mid_dim = h_dims[0]
        self.init_conv = nn.Conv1d(z_dim, mid_dim, 7, padding=3)
        self.mid_block1 = res_block(mid_dim, mid_dim, dropout=dropout)
        self.mid_attn = Residual(
            PreNorm(mid_dim, attn_block(mid_dim, attn_heads, attn_dim_head))
        )
        self.mid_block2 = res_block(mid_dim, mid_dim, dropout=dropout)

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
                                    dropout=dropout,
                                )
                                for i in range(num_res_blocks)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                Residual(
                                    PreNorm(
                                        dim_out,
                                        attn_block(dim_out, attn_heads, attn_dim_head),
                                    )
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
        return x


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
        attn_heads=8,
        attn_dim_head=32,
        commitment_weight=0.25,
        dropout=0.5,
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
            dropout=dropout,
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
            dropout=dropout,
        )

        self.vq = VectorQuantize(
            dim=emb_dim,
            codebook_size=n_emb,
            channel_last=False,
            commitment_weight=commitment_weight,
        )
        self.quant_conv = nn.Conv1d(z_dim, emb_dim, 1)
        self.post_quant_conv = nn.Conv1d(emb_dim, z_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.vq(h)

    def decode(self, quantized):
        quantized = self.post_quant_conv(quantized)
        return self.decoder(quantized)

    def forward(self, x, return_indices=False):
        quantized, indices, commit_loss = self.encode(x)
        decoded = self.decode(quantized)
        if return_indices:
            return decoded, commit_loss, indices
        else:
            return decoded, commit_loss
