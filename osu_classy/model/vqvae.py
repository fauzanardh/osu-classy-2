import torch
from torch import nn
from torch.autograd import Function


def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print(f"Skipping weights init for {class_name}")


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            emb_size = codebook.size(1)

            inputs_flatten = inputs.view(-1, emb_size)

            distances = torch.addmm(
                torch.sum(codebook**2, dim=1)
                + torch.sum(inputs_flatten**2, dim=1, keepdim=True),
                inputs_flatten,
                codebook.t(),
                alpha=-2.0,
                beta=1.0,
            )

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs.size()[:-1])

            ctx.mark_non_differentiable(indices)

        return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "Trying to call `.grad()` on graph containing "
            "`VectorQuantization`. The function `VectorQuantization` "
            "is not differentiable. Use `VectorQuantizationStraightThrough` "
            "if you want a straight-through estimator of the gradient."
        )


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = VectorQuantization.apply(inputs, codebook)
        indices_flatten = indices.view(-1)

        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices)
        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)

        codes = codes_flatten.view_as(inputs)
        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:
            indices_flatten, codebook = ctx.saved_tensors
            emb_size = codebook.size(1)

            grad_output_flatten = grad_output.contiguous().view(-1, emb_size)

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices_flatten, grad_output_flatten)

        return grad_inputs, grad_codebook


class VQEmbedding(nn.Module):
    def __init__(self, n_emb, emb_dim):
        super().__init__()

        self.embedding = nn.Embedding(n_emb, emb_dim)
        self.embedding.weight.data.uniform_(-1 / n_emb, 1 / n_emb)

    def forward(self, z_e_x):
        # BCL -> BLC
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        latents = VectorQuantization.apply(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        # BCL -> BLC
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()

        z_q_x_, indices = VectorQuantizationStraightThrough.apply(
            z_e_x_, self.embedding.weight.detach()
        )
        # BLC -> BCL
        z_q_x = z_q_x_.permute(0, 2, 1).contiguous()

        z_q_x_bar_flatten = torch.index_select(
            self.embedding.weight, dim=0, index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        # BLC -> BCL
        z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()

        return z_q_x, z_q_x_bar


class ConvNextBlock(nn.Module):
    def __init__(self, h_dim, mult=2, groups=1):
        super().__init__()

        self.ds_conv = nn.Conv1d(
            h_dim, h_dim, 7, padding=3, groups=h_dim, padding_mode="reflect"
        )
        self.net = nn.Sequential(
            nn.GroupNorm(1, h_dim),
            nn.Conv1d(
                h_dim,
                h_dim * mult,
                7,
                stride=1,
                padding=3,
                padding_mode="reflect",
                groups=groups,
            ),
            nn.SiLU(),
            nn.GroupNorm(1, h_dim * mult),
            nn.Conv1d(
                h_dim * mult,
                h_dim,
                7,
                stride=1,
                padding=3,
                padding_mode="reflect",
                groups=groups,
            ),
        )
        self.apply(init_weights)

    def forward(self, x):
        h = self.ds_conv(x)
        h = self.net(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, 7, padding=3, padding_mode="reflect"),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, 7, padding=3, padding_mode="reflect"),
            nn.BatchNorm1d(h_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_emb):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, h_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, 7, stride=2, padding=3),
            ConvNextBlock(h_dim),
            ConvNextBlock(h_dim),
        )
        self.codebook = VQEmbedding(n_emb, h_dim)
        self.decoder = nn.Sequential(
            ConvNextBlock(h_dim),
            ConvNextBlock(h_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, h_dim, 4, stride=2, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim, out_dim, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.apply(init_weights)

    def encode(self, x, a):
        x = torch.cat([x, a], dim=1)
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 2, 1)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
