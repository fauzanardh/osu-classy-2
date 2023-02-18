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
        unknown_index="random",
        legacy=True,
    ):
        super().__init__()

        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta
        self.unknown_index = unknown_index
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_emb, 1 / self.n_emb)

        self.re_embed = n_emb

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1, "Expected at least 2 dimensions"
        inds = inds.view(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.view(ishape)

    def forward(self, z):
        # BCL -> BLC
        z = z.permute(0, 2, 1).contiguous()
        z_flatten = z.view(-1, self.emb_dim)

        distances = torch.addmm(
            (
                torch.sum(z_flatten**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
            ),
            z_flatten,
            self.embedding.weight.t(),
            alpha=-2.0,
            beta=1.0,
        )

        min_encoding_indices = torch.argmin(distances, dim=-1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
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
        self.conv = torch.nn.Conv1d(in_channels, in_channels, 7, padding=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="linear")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, 7, stride=2, padding=3)

    def forward(self, x):
        x = self.conv(x)
        return x


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

        if in_dim != out_dim:
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


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        z_dim,
        double_z=True,
        dim_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.5,
    ):
        super().__init__()

        in_dim_mult = (1,) + tuple(dim_mult)
        self.init_conv = nn.Conv1d(in_dim, h_dim, 7, padding=3)

        # down
        self.down = nn.ModuleList()
        for i in range(len(dim_mult)):
            block = nn.ModuleList()
            block_in_dim = h_dim * in_dim_mult[i]
            block_out_dim = h_dim * dim_mult[i]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in_dim, block_out_dim, dropout))
                block_in_dim = block_out_dim

            down = nn.Module()
            down.block = block
            if i != len(dim_mult) - 1:
                down.downsample = Downsample(block_out_dim)

            self.down.append(down)

        # middle
        self.middle = nn.Sequential(
            ResnetBlock(block_in_dim, block_in_dim, dropout),
            ResnetBlock(block_in_dim, block_in_dim, dropout),
        )

        # end
        self.norm = nn.GroupNorm(1, block_in_dim)
        self.conv_out = nn.Conv1d(
            block_in_dim, 2 * z_dim if double_z else z_dim, 7, padding=3
        )

    def forward(self, x):
        # downsample
        h = self.init_conv(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
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
        double_z=True,
        dim_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.5,
    ):
        super().__init__()

        dim_mult = tuple(reversed(dim_mult))
        in_dim_mult = tuple(reversed((1,) + tuple(dim_mult)))
        block_in_dim = h_dim * in_dim_mult[0]

        self.init_conv = nn.Conv1d(
            z_dim * 2 if double_z else z_dim, block_in_dim, 7, padding=3
        )

        # middle
        self.middle = nn.Sequential(
            ResnetBlock(block_in_dim, block_in_dim, dropout),
            ResnetBlock(block_in_dim, block_in_dim, dropout),
        )

        # up
        self.up = nn.ModuleList()
        for i in range(len(dim_mult)):
            block = nn.ModuleList()
            block_out_dim = h_dim * dim_mult[i]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in_dim, block_out_dim, dropout))
                block_in_dim = block_out_dim

            up = nn.Module()
            up.block = block
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
            if hasattr(up, "upsample"):
                h = up.upsample(h)

        # end
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        h = torch.sigmoid(h)
        return h


class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim, n_emb, emb_dim, beta=0.25, double_z=True):
        super().__init__()

        self.encoder = Encoder(in_dim, h_dim, z_dim, double_z)
        self.decoder = Decoder(in_dim, h_dim, z_dim, double_z)
        self.quantize = VectorQuantizer(n_emb, emb_dim, beta=beta)
        self.quant_conv = nn.Conv1d(z_dim * 2 if double_z else z_dim, emb_dim, 1)
        self.post_quant_conv = nn.Conv1d(emb_dim, z_dim * 2 if double_z else z_dim, 1)

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
        if return_pred_indices:
            return decoded, loss, ind
        return decoded, loss
