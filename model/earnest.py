from model import common
import torch
import torch.nn as nn
import model.MPNCOV as MPNCOV


def make_model(args, parent=False):
    return EARNEST(args)


class NONLocalBlock1D(common._NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(common._NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]

        # MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub)  # Global Covariance pooling layer
        # Matrix square root layer( including pre-norm,Newton-Schulz iter  and post-com. with 5 iteration)
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat, 5)

        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)

        y_cov = self.conv_du(cov_mat_sum)

        return y_cov * x


# self-attention+ channel attention module
class NonlocalRL(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, sub_sample=False, bn_layer=True):
        super(NonlocalRL, self).__init__()

        self.non_local = (
            NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))

    def forward(self, x):

        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2):
        super(RB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0

        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )

        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y


## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(LSRAG, self).__init__()
        ##
        self.rcab = nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, act=act, res_scale=res_scale) for _ in
                                   range(n_resblocks)])

        self.soca = (SOCA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.gamma = 0.2

    def forward(self, x):
        residual = x

        # share-source skip connection

        for i, l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)

        x = x + residual

        return x
        ##


# Second-order Channel Attention Network (SAN)
class EARNEST(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EARNEST, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]

        if args.act == "relu":
            act = nn.ReLU(True)
        elif args.act == "xunit":
            act = common.xUnit(n_feats)
        elif args.act == "lrelu":
            act = nn.LeakyReLU()
        elif args.act == "prelu":
            act = nn.PReLU()

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # share-source skip connection
        ##
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups

        #                                       SSRG
        self.RG = nn.ModuleList([LSRAG(conv, n_feats, kernel_size, reduction,
                                       act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in
                                 range(n_resgroups)])
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.non_local = NonlocalRL(in_feat=n_feats, inter_feat=n_feats // 8, sub_sample=False,
                                    bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        ### Non-locally Enhanced Residual Group (NLRG)
        # add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # share-source residual gruop
        for i, l in enumerate(self.RG):
            xx = l(xx) + self.gamma * residual
            # xx = self.gamma*xx + residual

        # add nonlocal
        res = self.non_local(xx)

        res = res + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))