import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        
    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x, temb, fa, fb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb, fa, fb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
            
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.cross_attention = CrossAttentionSpatial(out_ch, emb_dim=128)
        self.conbine = nn.Sequential(
            nn.Conv2d(3 * out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
        )


    def forward(self, x, temb, fa, fb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)

        z1 = self.cross_attention(h, fa)
        z1 = self.cross_attention(z1, fb)
        z2 = self.cross_attention(h, fb)
        z2 = self.cross_attention(z2, fa)
        z = self.conbine(torch.cat([z1, z2, z1 + z2], dim=1)) + h
        
        return z


class CrossAttentionSpatial(nn.Module):
    def __init__(self, channel, emb_dim):
        super().__init__()
        self.norm_x = nn.GroupNorm(32, channel)

        self.norm_cond = nn.GroupNorm(32, emb_dim)
        self.proj_q = nn.Conv2d(channel, channel, 1, stride=1, padding=0)
        self.proj_k1 = nn.Conv2d(emb_dim, channel, 1, stride=1, padding=0)
        self.proj_v1 = nn.Conv2d(emb_dim, channel, 1, stride=1, padding=0) 
        self.scale = math.sqrt(channel)

    def forward(self, x, condA):
        B, C, H, W = x.shape

        h = self.norm_x(x)
        h1 = self.norm_cond(condA)

        q = self.proj_q(h)  # Shape: (B, C, H, W)

        k1 = self.proj_k1(h1)  # Shape: (B, C, H1, W1)
        v1 = self.proj_v1(h1)  # Shape: (B, C, H1, W1)
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # Shape: (B, H*W, C)

        k1 = k1.view(B, C, -1)  # Shape: (B, C, H1*W1)
        w1 = torch.bmm(q, k1) / self.scale  # Shape: (B, H*W, H1*W1)
        w1 = F.softmax(w1, dim=-1)
        v1 = v1.permute(0, 2, 3, 1).view(B, -1, C)  # Shape: (B, H1*W1, C)

        h1 = torch.bmm(w1, v1)  # Shape: (B, H*W, C)
        h1 = h1.view(B, H, W, C).permute(0, 3, 1, 2)  # Shape: (B, C, H, W)
        return h1


class APFM(nn.Module):
    def __init__(self, in_channels):
        super(APFM, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels * 4, 2 * in_channels, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.GroupNorm(32, 2 * in_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.GroupNorm(32, in_channels)
        
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.GroupNorm(32, in_channels)
        self.silu2 = nn.SiLU()
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm4 = nn.GroupNorm(32, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        # Concatenate along the channel dimension
        xy = torch.cat((x, y), dim=1)
        # Global Average Pooling and Global Max Pooling
        xy_avg = self.global_avg_pool(xy)
        xy_max = self.global_max_pool(xy)
        # Concatenate the pooled features
        xy_pooled = torch.cat((xy_avg, xy_max), dim=1)
        # First Convolution
        xy_conv1 = self.conv1(xy_pooled)
        xy_norm1 = self.norm1(xy_conv1)
        xy_silu = self.silu(xy_norm1)
        
        # Second Convolution
        xy_conv2 = self.conv2(xy_silu)
        xy_norm2 = self.norm2(xy_conv2)

        xy_conv3 = self.conv3(xy)
        xy_norm3 = self.norm3(xy_conv3)
        xy_silu2 = self.silu2(xy_norm3)
        xy_conv4 = self.conv4(xy_silu2)
        xy_norm4 = self.norm4(xy_conv4)
        
        # Sigmoid Activation
        xy_sigmoid = self.sigmoid(xy_norm2 + xy_norm4)

        # Element-wise multiplication and addition
        out = x * xy_sigmoid + y * (1 - xy_sigmoid)
        return out


class ContentEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim):
        super(ContentEncoder, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_dim, embed_dim // 4, kernel_size=3, stride=1, padding=1),

            nn.GroupNorm(32, embed_dim // 4),
            Swish(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1),

            nn.GroupNorm(32, embed_dim // 2),
            Swish(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.fusion = APFM(in_channels=embed_dim)
        
        self.enhance = nn.Sequential(
            nn.GroupNorm(32, embed_dim),
            Swish(),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, embed_dim // 2),
            Swish(),
            nn.Conv2d(embed_dim // 2, out_dim, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x, y):
        hx = self.conv(x)
        hy = self.conv(y)
        h = self.fusion(hx, hy)
        h = self.enhance(h)
        return h

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.layers, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class SemanticEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(SemanticEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1),

            nn.GroupNorm(32, 32),
            Swish(),
            nn.Conv2d(32, embed_dim // 4, kernel_size=3, stride=2, padding=1),

            nn.GroupNorm(32, embed_dim // 4),
            Swish(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),

            nn.GroupNorm(32, embed_dim // 2),
            Swish(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),

            nn.GroupNorm(32, embed_dim),
            Swish(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        )

        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout)

    def forward(self, x):
        h = self.conv(x)
        B, C, H, W = h.shape
        h = h.flatten(2).permute(2, 0, 1)  # (B, C, H, W) -> (H*W, B, embed_dim)
        h = self.transformer_encoder(h)
        h = h.permute(1, 2, 0)  # (H*W, B, C) -> (B, C, H*W)
        h = h.reshape(B, -1, H, W)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.content_encoder = ContentEncoder(in_dim=3, embed_dim=128, out_dim=64)
        self.encoder = SemanticEncoder(3, 128, 8, 4, dropout=dropout)
        
        self.proj_in = self.conbine = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(128, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, t, fa=None, fb=None, f_f=None):
        if fa is None and fb is None and f_f is None:
            xa = x[:, :3, :, :]
            xb = x[:, 3:6, :, :]
            fa = self.encoder(xa)
            fb = self.encoder(xb)
            f_f = self.content_encoder(xa, xb)
        x_in_proj = self.proj_in(x[:, 6:, :, :])
        x_in = torch.cat([f_f, x_in_proj], dim=1)
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x_in)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, fa, fb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, fa, fb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, fa, fb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 3, 2], attn=[2], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 9, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(model)
    print(y.shape)


