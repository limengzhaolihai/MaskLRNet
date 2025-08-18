import torch.nn as nn
import math
from model.FourierUnit import FourierUnit
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite

import torch
import torch.nn.functional as F
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self



class RepVGGDZ(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv1xN_Nx1_Depthwise(ed, ed, stride=1)  # 使用深度卷积替代普通卷积
        self.diagonal_conv = DiagonalwiseRefactorization(ed, stride=1)  # 中心感受野
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        conv_output = self.conv(x)
        diagonal_output = self.diagonal_conv(x)
        combined_output = conv_output + diagonal_output
        residual_output = combined_output + x
        return self.bn(residual_output)


class Conv1xN_Nx1_Depthwise(nn.Module):
    """
    替代普通 Conv1x3 -> Conv3x1 的结构，使用深度卷积减少参数
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1),
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0),
                      groups=in_channels, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_mask(in_channels):
    mask = torch.zeros((in_channels, 1, 3, 3))
    for i in range(in_channels):
        mask[i, 0, 1, 1] = 1.0
    return mask


# class DiagonalwiseRefactorization(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, groups=None):
#         super().__init__()
#         if groups is None:
#             groups = max(in_channels // 32, 1)
#         assert in_channels % groups == 0
#         self.groups = groups
#         self.channels = in_channels // groups
#         self.stride = stride

#         self.splitw = nn.ModuleList([
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=stride, padding=1, groups=self.channels, bias=False)
#             for _ in range(groups)
#         ])
#         self.masks = [get_mask(self.channels) for _ in range(groups)]

#         for i, conv in enumerate(self.splitw):
#             with torch.no_grad():
#                 conv.weight *= self.masks[i]

#     def forward(self, x):
#         splitx = torch.split(x, self.channels, dim=1)
#         outputs = []
#         for i, x_group in enumerate(splitx):
#             out = self.splitw[i](x_group)
#             outputs.append(out)
#         return F.relu(torch.cat(outputs, dim=1))


class DiagonalwiseRefactorization(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride

        # 单个 depthwise 3x3 卷积
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        # 对每个通道的卷积核施加掩码
        mask = get_mask(in_channels)  # shape: [C,1,3,3]
        with torch.no_grad():
            self.conv.weight *= mask

    def forward(self, x):
        out = self.conv(x)
        return F.relu(out)
    
    
    
# class InceptionTokenMixer(nn.Module):
#     def __init__(self, in_channels, out_channels=None, stride=1):
#         super().__init__()
#         if out_channels is None:
#             out_channels = in_channels

#         groups = max(in_channels // 32, 1)
        
#         self.branch_1x3_3x1 = Conv1xN_Nx1(in_channels, out_channels, stride=stride)
#         self.branch_diag = DiagonalwiseRefactorization(in_channels, out_channels, stride=stride, groups=groups)

#         # self.adjust_diag = nn.Conv2d(out_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         out_a = self.branch_1x3_3x1(x)
#         out_b = self.branch_diag(x)
#         # out_b = self.adjust_diag(out_b)

#         # 取平均或相加或concat
#         out = out_a + out_b  # or torch.cat([out_a, out_b], dim=1) with proper adjustment
#         return out

class DWConv(nn.Module):
    """Depthwise + Pointwise"""
    def __init__(self, c1, c2, k=3, s=1, d=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, d*(k-1)//2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class Conv(nn.Module):
    """Standard convolution with optional activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        if p is None:
            p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DownSimperLite(nn.Module):
    """Lightweight downsampling module with DWConv and asymmetric paths."""
    def __init__(self, c1, c2):
        super().__init__()
        assert c2 % 4 == 0, "Output channels c2 must be divisible by 4"
        c_half = c2 // 2
        c_quarter = c2 // 4
        self.cv1 = DWConv(c1, c_half, k=3, s=2)  # Path 1: Depthwise downsample
        self.cv2 = DWConv(c1, c_half, k=1, s=1)  # Path 2: Depthwise identity

    def forward(self, x):
        x1 = self.cv1(x)                         # [B, c2//2, H/2, W/2]
        x_temp = self.cv2(x)                     # [B, c2//2, H, W]
        x2, x3 = x_temp.chunk(2, dim=1)          # [B, c2//4, H, W]
        x2 = F.max_pool2d(x2, 3, 2, 1)           # [B, c2//4, H/2, W/2]
        x3 = F.avg_pool2d(x3, 3, 2, 1)           # [B, c2//4, H/2, W/2]
        return torch.cat((x1, x2, x3), dim=1)    # [B, c2, H/2, W/2]



class MaskLRNetBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(MaskLRNetBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        # assert(hidden_dim == 2 * inp)
        assert(hidden_dim >= 2 * inp)
        if stride == 2:#下采样的时候
            self.token_mixer = nn.Sequential(
                DownSimperLite(inp, oup),  
                # Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                # #InceptionTokenMixer(inp, out_channels=inp, stride=stride),
                # # DualBranchModule(inp, inp, stride),
                # SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                # Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            print("end inception")
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                # InceptionTokenMixer(inp, out_channels=inp, stride=stride),
                RepVGGDZ(inp),
                # DiagonalwiseRefactorization(inp, inp, stride=1),  # 加入对角线卷积
                # InceptionTokenMixer(inp, out_channels=inp, stride=stride),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
    

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels  # ✅ 添加这一行，解决 AttributeError

        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.GELU() if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.GELU() if relu else nn.Identity()
        )

    def forward(self, x):
        primary_out = self.primary_conv(x)
        cheap_out = self.cheap_operation(primary_out)
        out = torch.cat([primary_out, cheap_out], dim=1)
        return out[:, :self.out_channels, :, :]




class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True, use_ghost=True):
        super().__init__()
        self.distillation = distillation
        self.use_ghost = use_ghost

        if use_ghost:
            self.ghost = GhostModule(dim, dim, ratio=4, dw_size=1, relu=True)  # 极小化参数
        else:
            self.ghost = nn.Identity()

        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.ghost(x.unsqueeze(-1).unsqueeze(-1))  # 变成 [B,C,1,1]
        x = x.flatten(1)  # 变回 [B,C]

        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x


#ghost 解决亘余计算 cheap calucate 所




class MaskLRNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(MaskLRNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            DiagonalwiseRefactorization(input_channel // 2, stride=1),

            # InceptionTokenMixer(input_channel,input_channel, stride=2),
            #nn.GELU(),
            FourierUnit(input_channel // 2, input_channel // 2),  # 添加傅里叶变换模块  延迟性测试的时候不用傅里叶变换
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        )
        
        
        

        layers = [patch_embed]
        # building inverted residual blocks
        block = MaskLRNetBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        
        
        
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        # ▼ 新增：用 GhostModule 作瓶颈 (如 512→512 或 320→320)
        # self.ghost_tail = GhostModule(input_channel, input_channel, ratio=2)
        self.classifier = Classfier(output_channel, num_classes, distillation)
       
    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        # x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # x = self.global_pool(x)       # [B, C, 1, 1]
        # # x = self.ghost_tail(x)        # [B, C, 1, 1]
        # x = x.flatten(1)    
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)# [B,C]
        x = self.classifier(x)
        return x

from timm.models import register_model




@register_model
def MaskLRNet(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],
        [3,   2,  40, 0, 0, 1],
        [3,   2,  80, 0, 0, 2],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 1, 2],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],
        [3,   2, 320, 1, 1, 1],
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def MaskLRNet_m0_9(pretrained=False, num_classes = 1000, distillation=False):#MaskLRNet*
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   4,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def MaskLRNet_m1_0(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  56, 1, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  112, 0, 0, 2],
        [3,   2,  112, 1, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  224, 0, 1, 2],
        [3,   2,  224, 1, 1, 1],
        [3,   2,  224, 0, 1, 1],
        [3,   2,  224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 448, 0, 1, 2],
        [3,   2, 448, 1, 1, 1],
        [3,   2, 448, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def MaskLRNet_m1_1(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)



@register_model
def MaskLRNet_m1_5(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)



@register_model
def MaskLRNet_m2_3(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 0, 2],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  320, 0, 1, 2],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 640, 0, 1, 2],
        [3,   2, 640, 1, 1, 1],
        [3,   2, 640, 0, 1, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]    
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)
@register_model
def checkpoint_142(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   4, 384, 0, 1, 2],
        [3,   4, 384, 1, 1, 1],
        [3,   4, 384, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)




@register_model
def checkpoint_best(pretrained=False, num_classes = 1000, distillation=False,**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   4,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return MaskLRNet(cfgs, num_classes=num_classes, distillation=distillation)