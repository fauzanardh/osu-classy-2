import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print(f"Skipping weights init for {class_name}")


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_emb,
        emb_dim,
        beta,
    ):
        super().__init__()

        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_emb, 1 / self.n_emb)

        self.re_embed = n_emb

    def forward(self, z):
        # BCL -> BLC
        z = z.permute(0, 2, 1).contiguous()
        z_flatten = z.view(-1, self.emb_dim)

        distances = (
            torch.sum(z_flatten**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("b d, d n -> b n", z_flatten, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(distances, dim=-1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        # BLC -> BCL
        z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q


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
        # b h l c -> b (h c) l
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(out.shape[0], -1, out.shape[-1])
        return out

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        out = self.attn(*(t.unflatten(1, (self.heads, -1)) for t in qkv))
        out = self.to_out(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm1 = nn.GroupNorm(1, in_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 7, padding=3)

        self.norm2 = nn.GroupNorm(1, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_dim, out_dim, 7, padding=3)

        if self.in_dim != self.out_dim:
            self.shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_dim != self.out_dim:
            x = self.shortcut(x)

        return x + h


class ConvNextBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult=2):
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
        use_conv_next=False,
        num_res_blocks=2,
        attn_heads=8,
        attn_dim_head=32,
        dropout=0.5,
    ):
        super().__init__()

        in_dim_mult = (1,) + tuple(dim_mult)
        self.init_conv = nn.Conv1d(in_dim, h_dim, 7, padding=3)

        res_block = ConvNextBlock if use_conv_next else ResnetBlock

        # down
        self.down = nn.ModuleList()
        for i in range(len(dim_mult)):
            block = nn.ModuleList()
            block_in_dim = h_dim * in_dim_mult[i]
            block_out_dim = h_dim * dim_mult[i]
            for _ in range(num_res_blocks):
                block.append(res_block(block_in_dim, block_out_dim, dropout))
                block_in_dim = block_out_dim

            down = nn.Module()
            down.block = block
            down.attn = Attention(block_out_dim, attn_heads, attn_dim_head)

            if i != len(dim_mult) - 1:
                down.downsample = Downsample(block_out_dim)

            self.down.append(down)

        # middle
        self.middle = nn.Sequential(
            res_block(block_in_dim, block_in_dim, dropout),
            Attention(block_in_dim, attn_heads, attn_dim_head),
            res_block(block_in_dim, block_in_dim, dropout),
        )

        # end
        self.norm = nn.GroupNorm(1, block_in_dim)
        self.conv_out = nn.Conv1d(block_in_dim, z_dim, 7, padding=3)

    def forward(self, x):
        # downsample
        h = self.init_conv(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            h = down.attn(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)

        # middle
        h = self.middle(h)

        # end
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        z_dim,
        dim_mult=(1, 2, 4, 8),
        use_conv_next=False,
        num_res_blocks=2,
        attn_heads=8,
        attn_dim_head=32,
        dropout=0.5,
    ):
        super().__init__()

        dim_mult = tuple(reversed(dim_mult))
        in_dim_mult = tuple(reversed((1,) + tuple(dim_mult)))
        block_in_dim = h_dim * in_dim_mult[0]

        self.init_conv = nn.Conv1d(z_dim, block_in_dim, 7, padding=3)

        res_block = ConvNextBlock if use_conv_next else ResnetBlock

        # middle
        self.middle = nn.Sequential(
            res_block(block_in_dim, block_in_dim, dropout),
            Attention(block_in_dim, attn_heads, attn_dim_head),
            res_block(block_in_dim, block_in_dim, dropout),
        )

        # up
        self.up = nn.ModuleList()
        for i in range(len(dim_mult)):
            block = nn.ModuleList()
            block_out_dim = h_dim * dim_mult[i]
            for _ in range(num_res_blocks):
                block.append(res_block(block_in_dim, block_out_dim, dropout))
                block_in_dim = block_out_dim

            up = nn.Module()
            up.block = block
            up.attn = Attention(block_out_dim, attn_heads, attn_dim_head)
            if i != 0:
                up.upsample = Upsample(block_out_dim)

            self.up.append(up)

        # end
        self.norm = nn.GroupNorm(1, block_in_dim)
        self.conv_out = nn.Conv1d(block_in_dim, in_dim, 7, padding=3)

    def forward(self, x):
        # upsample
        h = self.init_conv(x)

        # middle
        h = self.middle(h)

        # upsample
        for up in self.up:
            for block in up.block:
                h = block(h)
            h = up.attn(h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)

        # end
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim, n_emb, emb_dim, beta=0.25):
        super().__init__()

        self.encoder = Encoder(in_dim, h_dim, z_dim)
        self.decoder = Decoder(in_dim, h_dim, z_dim)
        self.quantize = VectorQuantizer(n_emb, emb_dim, beta=beta)
        self.quant_conv = nn.Conv1d(z_dim, emb_dim, 1)
        self.post_quant_conv = nn.Conv1d(emb_dim, z_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize(h)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        return self.quant_conv(h)

    def decode(self, quantized):
        quantized = self.post_quant_conv(quantized)
        return self.decoder(quantized)

    def decode_code(self, codebook):
        quantized_codebook = self.post_quant_conv(codebook)
        return self.decoder(quantized_codebook)

    def forward(self, x, return_pred_indices=False):
        quantized, loss, (_, _, ind) = self.encode(x)
        decoded = self.decode(quantized)
        decoded = torch.tanh(decoded)
        if return_pred_indices:
            return decoded, loss, ind
        return decoded, loss
