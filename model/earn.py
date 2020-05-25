import torch.nn as nn
from model import common
from model.stand_alone_attention import AttentionConv, AttentionLite


def make_model(args):
    return EARN(args)


# Direkt Residual Blocklardan olusuyor
class TrunkBranch(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()
        modules_body = [common.ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale),
                        common.ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)
        return tx


# channel attention ve pixel attentionin yeri nerede olsun
# define mask branch
class MaskBranch(nn.Module):
    """
    Changelog:
    Residual blocklar yerine CARN daki efficient residual blokklar konuldu, group convolution yapiyor ve relu kullaniyor
    """
    def __init__(self, conv, n_feat, kernel_size, act=nn.ReLU(True), reduction=16):
        super(MaskBranch, self).__init__()

        # MB_RB1 = [common.ResBlockEfficient(conv, n_feat, kernel_size, act=act)]
        MB_RB1 = [common.ResBlockEfficient(conv, n_feat, kernel_size, act=act, groups=n_feat)]

        MB_Down = [AttentionLite(n_feat, n_feat, 3, stride=1, padding=1),
                   nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1)]

        MB_RB2 = [common.ResBlockEfficient(conv, n_feat, kernel_size, act=act, groups=n_feat)]

        # MB_Up = [nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2)]
        MB_Up = [nn.PixelShuffle(2)]

        self.Up1x1conv = nn.Conv2d(int(n_feat/4), n_feat, 1, padding=0)

        MB_RB3 = [common.ResBlockEfficient(conv, n_feat, kernel_size, act=act, groups=n_feat)]

        Att_MB_1x1conv = [nn.Conv2d(n_feat, n_feat, 1, padding=0)]
        MB_sigmoid = [nn.Sigmoid()]

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*Att_MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_Up = self.Up1x1conv(x_Up)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        mx = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(mx)

        return mx


class ResAttModule(nn.Module):
    """
    Changelog:
    channel attention ve pixel attention konuldu mask ve trunk brache ayrilmadan once
    """
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, reduction=16):
        r"""define non-local/local  module
            Args:
                bias (bool): if true conv layer will have bias term.
                bn (bool): if true batch normalization will be used
                act (Module): activation function
                res_scale (int): residual connection scale
            """
        super(ResAttModule, self).__init__()

        Attention_prior = [common.CALayer(n_feat, reduction),
                           common.CALayer(n_feat, reduction, pix_att=True)]

        RA_RB1 = [common.ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        RA_TB = [TrunkBranch(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        RA_MB = [MaskBranch(conv, n_feat, kernel_size, act=act)]

        RA_tail = [common.ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale),
                   common.ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale)]

        self.Att_tail = nn.Sequential(*Attention_prior)
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)


    def forward(self, input):
        r"""define non-local/local  module
            Args:
                input (Tensor): input tensor
            """
        Prior_Attention = self.Att_tail(input)
        RA_RB1_x = self.RA_RB1(Prior_Attention)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class _ResGroup(nn.Module):
    """
    Changelog:
    body modulunden den convolution kaldirildi
    skip connection eklendi
    """

    def __init__(self, conv, n_feats, k_size,  bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(_ResGroup, self).__init__()
        modules_body = [
            ResAttModule(conv, n_feats, k_size, bias=bias, bn=bn, act=act, res_scale=res_scale) #conv layer eklenebilir, attention eklendiginde cok iyi sonuc vermedi
        ]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x): # skip connection eklenmezse 5 sn daha hizli ama daha buyuk training loss veriyor
        res = self.body(x)
        return res + x


# EfficientAttentionalResidualNetworkSuperResolution
class EARN(nn.Module):
    """
    Changelog:
    Non-local attention block kaldirildi, stand olan attention eklendi
    """
    def __init__(self, args, conv=common.default_conv):
        super(EARN, self).__init__()
        
        n_resgroup = args.n_resgroups
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K 1-800
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            _ResGroup(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) for _ in range(n_resgroup)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.sub_mean(x)
        feats_shallow = self.head(x)

        res = self.body(feats_shallow)
        res += feats_shallow

        res_main = self.tail(res)

        res_main = self.add_mean(res_main)

        return res_main  

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
