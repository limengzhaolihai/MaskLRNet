import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 确保输入是浮点类型
        x = x.to(torch.float32)
        
        batch, c, h, w = x.size()
        new_h = 2 ** int(torch.ceil(torch.log2(torch.tensor(h))))
        new_w = 2 ** int(torch.ceil(torch.log2(torch.tensor(w))))
        
        if h != new_h or w != new_w:
            pad_h = new_h - h
            pad_w = new_w - w
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # 对输入进行傅里叶变换
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.real(ffted).unsqueeze(-1)
        x_fft_imag = torch.imag(ffted).unsqueeze(-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        
        # 对傅里叶变换后的特征进行卷积操作
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        # 将处理后的特征重新调整为复数形式
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        
        # 进行逆傅里叶变换
        output = torch.fft.irfft2(ffted, s=(new_h, new_w), norm='ortho')
        
        # 如果进行了填充，裁剪回原始尺寸
        if h != new_h or w != new_w:
            output = output[:, :, :h, :w]
        
        return output
