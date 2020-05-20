import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return RNAN(args)


class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, k_size, nl_att=False, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(_ResGroup, self).__init__()
        modules_body = [
            common.ResAttModule(conv, n_feats, k_size, nl_att=nl_att, bias=bias, bn=bn, act=act, res_scale=res_scale),
            conv(n_feats, n_feats, k_size)
        ]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res


class RNAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RNAN, self).__init__()
        
        n_resgroup = args.n_resgroups
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K 1-800
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        modules_body_nl_low = [
            _ResGroup(conv, n_feats, kernel_size, nl_att=True, act=act, res_scale=args.res_scale)
        ]

        modules_body = [
            _ResGroup(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) for _ in range(n_resgroup - 2)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_body_nl_high = [
            _ResGroup(conv, n_feats, kernel_size, nl_att=True, act=act, res_scale=args.res_scale)
        ]

        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, args.n_colors, kernel_size)
                        ]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.sub_mean(x)
        feats_shallow = self.head(x)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)
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
