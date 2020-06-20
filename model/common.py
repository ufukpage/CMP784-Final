import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-torch.mul(x, x))


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def gram_matrix_v2(x):
    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class xUnit(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, skernel_size=9):
        super(xUnit, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=((kernel_size-1)//2), bias=True))
        self.module = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=skernel_size, stride=1, padding=((skernel_size-1)//2),
                      groups=out_channels),
            nn.BatchNorm2d(out_channels),
            Gaussian())

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.module(x1)
        x = torch.mul(x1, x2)
        return x


class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeAndExcitationBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelDescriptorLayer(nn.Module):
    def __init__(self):
        super(ChannelDescriptorLayer, self).__init__()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
        x_mean = spatial_sum / (x.size(2) * x.size(3))
        x_variance = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
        return x_variance.pow(0.5), x_mean


class AdaptivelyScaledCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AdaptivelyScaledCALayer, self).__init__()

        self.local_channel_descriptors = ChannelDescriptorLayer()

        self.saeb_mean = SqueezeAndExcitationBlock(channel, reduction=reduction)
        self.saeb_std = SqueezeAndExcitationBlock(channel, reduction=reduction)

        self.small_descriptor_bottleneck = nn.Sequential(nn.Conv2d(2*channel, 1*channel, 1),
                                                         nn.ReLU(inplace=True))

        self.saeb_final = SqueezeAndExcitationBlock(channel, reduction=reduction)

        self.gating_function = nn.Sigmoid()

    def forward(self, x):

        std_des, mean_des = self.local_channel_descriptors(x)

        # refined descriptors
        ref_std_des = self.saeb_std(std_des)
        ref_mean_des = self.saeb_mean(mean_des)

        # descriptor fusion
        fused_des = torch.cat((ref_std_des, ref_mean_des), 1)

        # descriptor bottleneck
        fused_des = self.small_descriptor_bottleneck(fused_des)

        # final mask
        fused_des = self.saeb_final(fused_des)
        mask = self.gating_function(fused_des)

        return x * mask


# Channel/Pixel Based Attention (CA) Layer
class CALayer(nn.Module):

    """
    if pix_att is True then it does not use avg pooling and, it works as pixel attention.
    if contrast_aware is True then it uses summation of average and sta of channel instead of average .
    """
    def __init__(self, channel, reduction=16, contrast_aware=False, pix_att=False):
        super(CALayer, self).__init__()

        self.pix_att = pix_att
        self.contrast_aware = contrast_aware

        if contrast_aware:
            self.local_descriptor = CALayer.rescaled_contrast_layer

        if not pix_att and not contrast_aware:
            # global average pooling: feature --> point
            self.local_descriptor = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_att = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    @staticmethod
    def rescaled_contrast_layer(F):
        assert (F.dim() == 4)
        F_mean = mean_channels(F)
        F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
        # return F_mean / F_variance.pow(0.5)
        # return - F_mean + F_variance
        return -F_mean / F_variance.pow(0.5) + F_variance.pow(0.5)

    def forward(self, x):
        if not self.pix_att or self.contrast_aware:
            y = self.local_descriptor(x)
            y = self.conv_att(y)
        else:
            y = self.conv_att(x)
        return x * y


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()

        m = [conv(n_feats, n_feats, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return F.relu(res)


class ResBlockEfficient(nn.Module):
    """
    We use a similar approach to the MobileNet [16], but use group convolution instead of depthwise convolution. CARN
    """
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU(True), groups=4):
        super(ResBlockEfficient, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), groups=groups),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), groups=groups),
            act,
            nn.Conv2d(n_feats, n_feats, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z


# Direkt Residual Blocklardan olusuyor
class TrunkBranch(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


# define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(MaskBranchDownUp, self).__init__()

        MB_RB1 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        MB_Down = [nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1)]

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_Up = [nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2)]

        MB_RB3 = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)]

        MB_1x1conv = [nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True)]

        MB_sigmoid = [nn.Sigmoid()]

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


class ResAttModule(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, nl_att=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        r"""define non-local/local  module
            Args:
                nl_att (bool): if true non local attention will be used
                bias (bool): if true conv layer will have bias term.
                bn (bool): if true batch normalization will be used
                act (Module): activation function
                res_scale (int): residual connection scale
            """
        super(ResAttModule, self).__init__()
        RA_RB1 = [ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        RA_TB = [TrunkBranch(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        RA_MB = [NonLocalBlock2D(n_feat, n_feat // 2)] if nl_att else []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        r"""define non-local/local  module
            Args:
                input (Tensor): input tensor
            """
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx



# non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size, C, H, W = x.shape

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        # return f
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
