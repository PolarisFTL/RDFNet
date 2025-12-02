import torch
import torch.nn as nn
from nets.Common import Conv, SPPELAN
from nets.backbone import Backbone, Multi_Concat_Block

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=k)
        self.m2 = nn.AvgPool2d(kernel_size=k, stride=k)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        return self.up(x1 + x2)
    
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        transition_channels = 16
        block_channels      = 16
        panet_channels      = 16
        e                   = 1
        n                   = 2
        ids                 = [-1, -2, -3, -4]

        self.backbone   = Backbone(transition_channels, block_channels, n)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppelan                = SPPELAN(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2         = Conv(transition_channels * 16, transition_channels * 8)
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1         = Conv(transition_channels * 8, transition_channels * 4)
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1           = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2           = Conv(transition_channels * 8, transition_channels * 16, k=3, s=2)
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)

        self.pf = MP()

        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        if self.training:
            feat1, feat2, feat3, dehazing = self.backbone.forward(x)
        else:
            feat1, feat2, feat3 = self.backbone.forward(x)

        P5          = self.sppelan(feat3)
        P5_conv     = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4          = self.conv3_for_upsample1(P4)

        P4_conv     = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3          = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)
        P4 = self.pf(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)

        if self.training:
            return [out0, out1, out2, dehazing]
        else:
            return [out0, out1, out2]