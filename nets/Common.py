import torch
import torch.nn as nn
from thop import profile 

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.LeakyReLU(0.1, inplace=True)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.LeakyReLU(0.1, inplace=True)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
#lighting dehaze network
class LMDNet(nn.Module):

    def __init__(self):
        super(LMDNet, self).__init__()
        # mainNet Architecture
        self.AAM = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.AAM_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.AAM_2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 32, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.TA = TripletAttention(64)

        self.conv = Conv(64, 3, 3, 1)

        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, f1, f2, f3):
        t = self.AAM(f1) 
        f2 = self.AAM_1(f2)
        f3 = self.AAM_2(f3)
        x1 = f1
        x2 = torch.cat([f2, f3], dim=1)
        
        x = x1 + x2
        x = self.TA(x)
        x = self.conv(x)

        dehaze = ((x * t) - x + 1)

        out = self.up4(dehaze)
        out = self.relu(out)   

        return out
        
class TripletAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=nn.LeakyReLU(0.1, inplace=True)):  
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class GIE(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super(GIE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Step 1: Pixel Mean Squared
        x_mean = torch.mean(x, dim=(2, 3), keepdim=True) 
        epsilon = (x - x_mean) ** 2  
        # Step 2: Global Extraction
        epsilon_mean = torch.mean(epsilon, dim=(2, 3), keepdim=False)  
        epsilon_mean += self.epsilon
        # Step 3: Gamma Calculation (Global Extraction)
        gamma_t_c = epsilon / epsilon_mean.unsqueeze(-1).unsqueeze(-1)  
        sigmoid_gamma = torch.sigmoid(gamma_t_c) 
        output = x * sigmoid_gamma 
        return output
    
# Multi-branch Pooling Information Fusion Module
class MPIF(nn.Module):
    def __init__(self, c1, c2, c3, s=2, n=4, e=1, ids=[0]):
        super(MPIF, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        if s == 1:
            self.m1 = nn.MaxPool2d(kernel_size=3, stride=s, padding=1)
            self.m2 = nn.AvgPool2d(kernel_size=3, stride=s, padding=1)
        else:
            self.m1 = nn.MaxPool2d(kernel_size=2, stride=s)
            self.m2 = nn.AvgPool2d(kernel_size=2, stride=s)
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

        self.GIE = GIE(c1)

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        x = x1 + x2
        x_1 = self.cv1(x)
        x_1 = self.GIE(x_1)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]

        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
        
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))

        return out
    
class DilatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding, kernel_size):
        super(DilatedConvNet, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.dilated_conv(x)
        x = self.relu(x)

        return x
    
class SPPELAN(nn.Module):
    def __init__(self, c1, c2, c3=16):  
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = DilatedConvNet(c3, c3, kernel_size=3, padding=1, dilation=1)
        self.cv3 = DilatedConvNet(c3, c3, kernel_size=3, padding=2, dilation=2)
        self.cv4 = DilatedConvNet(c3, c3, kernel_size=3, padding=3, dilation=3)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
        


def print_model_flops_and_params(model, inputs):
    flops, params = profile(model, inputs=inputs)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")


if __name__ == "__main__":
    feat1 = torch.randn(1, 64, 160, 160)
    feat2 = torch.randn(1, 128, 80, 80) 
    feat3 = torch.randn(1, 256, 40, 40)  
    encoder = LMDNet()
    print_model_flops_and_params(encoder, (feat1, feat2, feat3))
    
