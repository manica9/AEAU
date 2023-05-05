import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from loss import ncc_loss
from loss import mse_loss
from loss import gradient_loss
from config import args
from ptflops import get_model_complexity_info
from einops import rearrange

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, alpha=0.1, stride=1, padding=1, bias=True, hasbn=False):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, bias=bias)
        self.activation = nn.LeakyReLU(alpha)
        self.bn = nn.BatchNorm3d(in_channels)
        self.hasbn = hasbn
        # self.sa = SpatialAttention(3)

    def forward(self, x):
        if self.hasbn:
            x = self.bn(x)
            # x = self.se(x)
        out = self.conv(x)
        out = self.activation(out)
        return out

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, alpha=0.1, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, bias=bias)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv(out))
        return out+x

class respath(nn.Module):
    def __init__(self, in_channels, out_channels, ksize1, ksize2, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, ksize1, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(in_channels, out_channels, ksize2, 1, 0, bias=bias)
        # self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = x
        for i in range(4):  # 或者 1
            conv1 = self.conv1(out)
            conv2 = self.conv2(out)
            out = conv1 + conv2
        return out

def affine_flow(W, b, len1, len2, len3):
    b = torch.reshape(b, [-1, 1, 1, 1, 3])
    xr = torch.arange(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, dtype=torch.float32)
    xr = torch.reshape(xr, [1, -1, 1, 1, 1])
    yr = torch.arange(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, dtype=torch.float32)
    yr = torch.reshape(yr, [1, 1, -1, 1, 1])
    zr = torch.arange(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, dtype=torch.float32)
    zr = torch.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = torch.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = torch.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = torch.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr.to(wx.device) * wx + yr.to(wy.device) * wy) + (zr.to(wz.device) * wz + b)

def det3x3(M):

    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return M[0][0] * M[1][1] * M[2][2] +  M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1] - \
           (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])

class Affine(nn.Module):
    def __init__(self, in_ch=2, out_ch=16):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(conv_block(in_ch, out_ch, 3, stride=1, padding=1, hasbn=True))  # 0

        self.conv.append(conv_block(out_ch, out_ch * 2, 3, stride=2, padding=1, hasbn=True))  # 1
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1, hasbn=True))  # 3 32 32 32 32
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1, hasbn=True))  # 5 32 16 16 16
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1, hasbn=True))  # 7 32 8 8 8
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1, hasbn=True))  # 9 32 4 4 4
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))  #

        self.conv.append(convblock(out_ch * 2, 9, (5, 6, 5), stride=1, padding=0))  # 1 9 1 1 1
        self.conv.append(convblock(out_ch * 2, 3, (5, 6, 5), stride=1, padding=0))

    def forward(self, x):
        conv1 = self.conv[0](x)
        conv2 = self.conv[1](conv1)
        conv3 = self.conv[2](conv2)
        conv4 = self.conv[3](conv3)
        conv5 = self.conv[4](conv4)
        conv6 = self.conv[5](conv5)
        conv7 = self.conv[6](conv6)
        conv8 = self.conv[7](conv7)
        conv9 = self.conv[8](conv8)
        conv10 = self.conv[9](conv9)
        conv11 = self.conv[10](conv10)
        conv12w = self.conv[11](conv11)
        conv12b = self.conv[12](conv11)

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = torch.reshape(conv12w, [-1, 3, 3])
        b = torch.reshape(conv12b, [-1, 3])
        A = W + torch.repeat_interleave(torch.from_numpy(np.array(I)).float().to(W.device), W.shape[0], dim=0)
        sx, sy, sz = x.shape[2:5]  # torch tensor = [B, C, H, W, D]
        det = det3x3(A)
        det_loss = torch.norm(det.float() - 1.0) ** 2 / 2
        flow = affine_flow(W, b, sx, sy, sz).permute(0, 4, 1, 2, 3)
        return {'flow': flow, 'W': W, 'b': b, 'det_loss': det_loss}

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w, d = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)

        out = self.project_out(out)
        return out

class AEAU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AEAU, self).__init__()
        self.conv = nn.ModuleList()
        self.respath = nn.ModuleList()
        self.att = Attention(out_ch * 2, 1, False)
        self.conv.append(conv_block(in_ch, out_ch, 3, stride=1, padding=1))
        self.respath.append(respath(out_ch, out_ch, 3, 1))


        self.conv.append(conv_block(out_ch, out_ch * 2, 3, stride=2, padding=1))
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))
        self.respath.append(respath(out_ch * 2, out_ch * 2, 3, 1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1))
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))
        self.respath.append(respath(out_ch * 2, out_ch * 2, 3, 1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1))
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))
        self.respath.append(respath(out_ch * 2, out_ch * 2, 3, 1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1))
        self.conv.append(res_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))
        self.respath.append(respath(out_ch * 2, out_ch * 2, 3, 1))

        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=2, padding=1))
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.up1 = nn.ConvTranspose3d(out_ch * 2, out_ch * 2, 4, stride=2, padding=1)

        self.conv.append(conv_block(out_ch * 4, out_ch * 2, 1, stride=1, padding=0))
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.up2 = nn.ConvTranspose3d(out_ch * 2, out_ch * 2, 4, stride=2, padding=1)

        self.conv.append(conv_block(out_ch * 4, out_ch * 2, 1, stride=1, padding=0))
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.up3 = nn.ConvTranspose3d(out_ch * 2, out_ch * 2, 4, stride=2, padding=1)

        self.conv.append(conv_block(out_ch * 4, out_ch * 2, 1, stride=1, padding=0))
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.up4 = nn.ConvTranspose3d(out_ch * 2, out_ch * 2, 4, stride=2, padding=1)

        self.conv.append(conv_block(out_ch * 4, out_ch * 2, 1, stride=1, padding=0))
        self.conv.append(conv_block(out_ch * 2, out_ch * 2, 3, stride=1, padding=1))

        self.up5 = nn.ConvTranspose3d(out_ch * 2, out_ch * 2, 4, stride=2, padding=1)

        self.conv.append(res_block(out_ch * 3, out_ch * 3, 3, stride=1, padding=1))
        self.conv.append(conv_block(out_ch * 3, 3, 3, stride=1, padding=1))

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        c1 = self.conv[0](x)
        c1b = self.respath[0](c1)

        c2 = self.conv[1](c1)
        c2b = self.respath[1](c2)

        c3 = self.conv[3](c2)
        c3b = self.respath[2](c3)

        c4 = self.conv[5](c3)
        c4b = self.respath[3](c4)

        c5 = self.conv[7](c4)
        c5b = self.respath[4](c5)

        c6 = self.conv[9](c5)
        c6 = self.att(c6)
        c7 = self.conv[10](c6)
        up1 = self.up1(c7)
        cat1 = torch.cat([up1, c5b], dim=1)
        c8 = self.conv[12](self.conv[11](cat1))
        c8 = self.conv[2](c8)

        up2 = self.up2(c8)
        cat2 = torch.cat([up2, c4b], dim=1)
        c9 = self.conv[14](self.conv[13](cat2))
        c9 = self.conv[4](c9)

        up3 = self.up3(c9)
        cat3 = torch.cat([up3, c3b], dim=1)
        c10 = self.conv[16](self.conv[15](cat3))
        c10 = self.conv[6](c10)

        up4 = self.up4(c10)
        cat4 = torch.cat([up4, c2b], dim=1)
        c11 = self.conv[18](self.conv[17](cat4))
        c11 = self.conv[8](c11)

        up5 = self.up5(c11)
        cat5 = torch.cat([up5, c1b], dim=1)
        c12 = self.conv[19](cat5)
        flow = self.conv[20](c12)
        return {'flow': flow}

class SpatialTransformer(nn.Module):  # 图像增强
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]   # 加了一个range
        grids = torch.meshgrid(vectors)  # 生成网格，可以用于生成坐标。
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

class TwoStage(nn.Module):
    def __init__(self, in_ch, out_ch, vol_size):
        super(TwoStage, self).__init__()

        self.affine = Affine()
        self.regunet = AEAU(in_ch, out_ch)
        self.stn = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        affine_moving_flow = self.affine(x)
        affine_mov = self.stn(src, affine_moving_flow['flow'])
        # 组合形变场
        reg_flow = self.regunet(affine_mov, tgt)
        flow = self.stn(affine_moving_flow['flow'], reg_flow['flow']) + reg_flow['flow']  # 混合形变场
        warp = self.stn(src, flow)
        return {'flow': flow, 'reg_flow': reg_flow['flow'], 'warp': warp, 'affine_flow': affine_moving_flow['flow'], 'affine_mov': affine_mov}


