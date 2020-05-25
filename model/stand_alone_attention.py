import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/in_paper_attention_augmented_conv/attention_augmented_conv.py
class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x




# https://github.com/20171130/AttentionLite/blob/master/model.py
class Sparse(nn.Module):
    """ #groups independent denses"""

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Sparse, self).__init__()
        assert (in_channels % groups == 0)
        assert (out_channels % groups == 0)
        in_w = in_channels // groups
        out_w = out_channels // groups
        self.layers = nn.ModuleList([nn.Conv2d(in_w, out_w, kernel_size=1, bias=bias) for i in range(groups)])
        self.groups = groups
        self.reset_parameters()

    def forward(self, x, cat=True):
        batch, channels, height, width = x.size()
        x = torch.chunk(x, self.groups, dim=1)

        out = [self.layers[i](x[i]) for i in range(self.groups)]
        # shape group tensor(b, out_w, h, w)

        if cat:
            out = torch.cat(out, dim=1)
            # (b, out_channels, h, w)

        return out

    def reset_parameters(self):
        for i in range(self.groups):
            init.kaiming_normal_(self.layers[i].weight, mode='fan_out', nonlinearity='relu')


class AttentionLite(nn.Module):
    """ Routes information among the densely connected groups and their neighbours via attention"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, heads=1):
        super(AttentionLite, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.heads = heads
        in_w = in_channels // groups
        out_w = out_channels // groups
        self.in_w = in_w
        self.out_w = out_w

        self.query_conv = Sparse(in_channels, out_channels * heads, groups)
        self.key_conv = Sparse(in_channels, out_channels, groups)
        self.value_conv = Sparse(in_channels, out_channels, groups)

        self.rel_h = nn.Parameter(torch.randn(out_channels // groups // 2, 1, 1, groups, kernel_size, 1),
                                  requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // groups // 2, 1, 1, groups, 1, kernel_size),
                                  requires_grad=True)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x, cat=False)
        k_out = self.key_conv(padded_x, cat=False)
        v_out = self.value_conv(padded_x, cat=False)
        # shape tensor(b, out_channels, h, w)

        q_out = torch.stack(q_out, dim=-1)
        k_out = torch.stack(k_out, dim=-1)
        v_out = torch.stack(v_out, dim=-1)
        # (b, out_w, h, w, group)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # shape (b, out_w, h, w, g, k, k)

        k_out_h, k_out_w = k_out.split(self.out_w // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        # positional embeddings

        k_out = k_out.contiguous().view(batch, 1, self.out_w, height, width, 1, -1)
        v_out = v_out.contiguous().view(batch, 1, self.out_w, height, width, 1, -1)
        q_out = q_out.view(batch, self.heads, self.out_w, height, width, self.groups, 1)
        # the last dim of k and v is groups*kernel**2

        out = q_out * k_out
        # (b, heads, out_w, h, w, g, g*k*k)
        out = out.sum(dim=2, keepdim=True)
        out = F.softmax(out, dim=-1)
        # out :(b, heads, 1, h, w, g, g*k*k)
        # v_out: (b, 1, out_w. h, w, 1, g*k*k)
        out = out * v_out
        # (b, heads, out_w, h, w, g, g*k*k)
        out = out.sum(dim=1)
        out = out.sum(dim=-1)
        # (b, out_w, h, w, g)
        out = out.transpose(1, -1)
        out = out.reshape(batch, -1, height, width)
        return out

    def reset_parameters(self):
        self.query_conv.reset_parameters()
        self.value_conv.reset_parameters()
        self.key_conv.reset_parameters()

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


# https://github.com/leaderj1001/Stand-Alone-Self-Attention/tree/a983f0f643632b1f2b7b8b27693182f22e9e574c
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        """ 
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        # out = (q_out * k_out).sum(dim=2)
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        """

        k_out = k_out.contiguous().view(batch, self.groups, height, width, -1, self.out_channels // self.groups)
        v_out = v_out.contiguous().view(batch, self.groups, height, width, -1, self.out_channels // self.groups)

        q_out = q_out.view(batch, self.groups, height, width, 1, self.out_channels // self.groups)

        out = torch.matmul(q_out, (k_out).transpose(-1, -2))
        out = F.softmax(out, dim=-1)

        out = torch.matmul(out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


import math
from torch.nn.modules.utils import _pair

# https://github.com/MerHS/SASA-pytorch
class SelfAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=True):
        super(SelfAttentionConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups # multi-head count

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0]))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_value = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)

        self.softmax = nn.Softmax(dim=3)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_value.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        init.normal_(self.relative_x, 0, 1)
        init.normal_(self.relative_y, 0, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x)
        vv = self.weight_value(x) # b, fc, ph, pw

        # b, fc, fh, fw
        win_q = vq[:, :, (kh-1)//2:ph-(kh//2):self.stride[0], (kw-1)//2:pw-(kw//2):self.stride[1]]

        win_q_b = win_q.view(b, self.groups, -1, fh, fw) # b, g, fc/g, fh, fw

        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2) # (b, g, x, fh, fw), (b, g, y, fh, fw)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x)) # b, fh, fw, kw
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y)) # b, fh, fw, kh

        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1]) # b, fc, fh, fw, kh, kw

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)  # b, fh, fw, kh, kw
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4) # add rel_x, rel_y
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)

        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, )) # (b, fc, fh, fw, kh, kw) -> (b, fc, fh, fw)

        if self.bias is not None:
            fin_v += self.bias

        return fin_v


class SAMixtureConv2d(nn.Module):
    """ spatially-aware SA / multiple value transformation for stem layer """
    def __init__(self, in_height, in_width, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, mix=4, bias=True):
        super(SAMixtureConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups # multi-head count
        self.mix = mix # weight mixture

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0]))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_values = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False) for _ in range(mix)])

        self.emb_x = nn.Parameter(torch.Tensor(out_channels // groups, in_width + 2 * self.padding[1])) # fc/g, pw
        self.emb_y = nn.Parameter(torch.Tensor(out_channels // groups, in_height + 2 * self.padding[0])) # fc/g, ph
        self.emb_m = nn.Parameter(torch.Tensor(mix, out_channels // groups)) # m, fc/g

        self.softmax = nn.Softmax(dim=3)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out', nonlinearity='relu')
        for wv in self.weight_values:
            init.kaiming_normal_(wv.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        init.normal_(self.relative_x, 0, 1)
        init.normal_(self.relative_y, 0, 1)
        init.normal_(self.emb_x, 0, 1)
        init.normal_(self.emb_y, 0, 1)
        init.normal_(self.emb_m, 0, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x) # b, fc, fh, fw

        # b, fc, fh, fw
        win_q = vq[:, :, (kh-1)//2:ph-(kh//2):self.stride[0], (kw-1)//2:pw-(kw//2):self.stride[1]]

        win_q_b = win_q.view(b, self.groups, -1, fh, fw) # b, g, fc/g, fh, fw

        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2) # (b, g, x, fh, fw), (b, g, y, fh, fw)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x)) # b, fh, fw, kw
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y)) # b, fh, fw, kh

        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1]) # b, fc, fh, fw, kh, kw

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)  # b, fh, fw, kh, kw
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4) # add rel_x, rel_y
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)

        # spatially aware mixture embedding
        p_abm_x = torch.einsum('mc,cw->mw', (self.emb_m, self.emb_x)).unsqueeze(1) # m, 1, pw
        p_abm_y = torch.einsum('mc,ch->mh', (self.emb_m, self.emb_y)).unsqueeze(2) # m, ph, 1
        p_abm = F.softmax(p_abm_x + p_abm_y, dim=0) # m, ph, pw

        vv = torch.stack([weight_value(x) for weight_value in self.weight_values], dim=0) # m, b, fc, ph, pw
        vv = torch.einsum('mbchw,mhw->bchw', (vv, p_abm)) # b, fc, ph, pw

        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, )) # (b, fc, fh, fw, kh, kw) -> (b, fc, fh, fw)

        if self.bias is not None:
            fin_v += self.bias

        return fin_v