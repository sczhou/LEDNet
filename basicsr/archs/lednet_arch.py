import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.lednet_submodules import *
from basicsr.utils.registry import ARCH_REGISTRY

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat         


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                Downsample(channels=in_channels, filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class BasicBlock_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_E, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = ResidualDownSample(in_channels, out_channels)

    def forward(self, x):
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out

class BasicBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        if self.mode is not None:
            x = self.reshape_conv(x)
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        return out


class BasicBlock_D_2Res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D_2Res, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
        )

    def forward(self, x):
        if self.mode is not None:
            x = self.reshape_conv(x)
        res1 = self.body1(x)
        out1 = res1 + x
        res2 = self.body2(out1)
        out2 = res2 + out1
        return out2


## Channel Attention (CA) Layer
class CurveCALayer(nn.Module):
    def __init__(self, channel, n_curve):
        super(CurveCALayer, self).__init__()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(channel, n_curve, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # clip the input features into range of [0,1]
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x-1)
        for i in range(self.n_curve):
            x = x + a[:,i:i+1]*x*(1-x)
        return x


@ARCH_REGISTRY.register()
class LEDNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 128], connection=False):
        super(LEDNet, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        self.connection = connection
        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, ch1, 3, stride=1, padding=1), nn.PReLU(),
            BasicBlock_E(ch1, ch2, mode='down'))
        self.E_block2 = BasicBlock_E(ch2, ch3, mode='down')
        self.E_block3 = BasicBlock_E(ch3, ch4, mode='down')

        self.side_out = nn.Conv2d(ch4, 3, 3, stride=1, padding=1)

        self.M_block1 = BasicBlock_E(ch4, ch4)
        self.M_block2 = BasicBlock_E(ch4, ch4)

        # dynamic filter
        ks_2d = 5
        self.conv_fac_k3 = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4* ks_2d**2, 1, stride=1))

        self.conv_fac_k2 = nn.Sequential(
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3* ks_2d**2, 1, stride=1))

        self.conv_fac_k1 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2* ks_2d**2, 1, stride=1))

        self.kconv_deblur = KernelConv2D(ksize=ks_2d, act=True)

        # curve
        self.curve_n = 3
        self.conv_1c = CurveCALayer(ch2, self.curve_n)
        self.conv_2c = CurveCALayer(ch3, self.curve_n)
        self.conv_3c = CurveCALayer(ch4, self.curve_n)

        self.PPM1 = PPM(ch2, ch2//4, bins=(1,2,3,6))
        self.PPM2 = PPM(ch3, ch3//4, bins=(1,2,3,6))
        self.PPM3 = PPM(ch4, ch4//4, bins=(1,2,3,6))

        self.D_block3 = BasicBlock_D_2Res(ch4, ch4)
        self.D_block2 = BasicBlock_D_2Res(ch4, ch3, mode='up')
        self.D_block1 = BasicBlock_D_2Res(ch3, ch2, mode='up')
        self.D_block0 = nn.Sequential(
            BasicBlock_D_2Res(ch2, ch1, mode='up'),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=1))     

    def forward(self, x, side_loss=False):
        # Dncoder
        e_feat1 = self.E_block1(x) #64 1/2
        e_feat1 = self.PPM1(e_feat1) 
        e_feat1 = self.conv_1c(e_feat1)

        e_feat2 = self.E_block2(e_feat1) #128 1/4
        e_feat2 = self.PPM2(e_feat2) 
        e_feat2 = self.conv_2c(e_feat2)

        e_feat3 = self.E_block3(e_feat2) #256 1/8
        e_feat3 = self.PPM3(e_feat3) 
        e_feat3 = self.conv_3c(e_feat3)

        if side_loss:
            out_side = self.side_out(e_feat3)

        # Mid
        m_feat = self.M_block1(e_feat3)
        m_feat = self.M_block2(m_feat)

        # Decoder
        d_feat3 = self.D_block3(m_feat) #256 1/8
        kernel_3  = self.conv_fac_k3(e_feat3)
        d_feat3 = self.kconv_deblur(d_feat3, kernel_3)
        if self.connection:
            d_feat3 = d_feat3 + e_feat3

        d_feat2 = self.D_block2(d_feat3)  #128 1/4
        kernel_2  = self.conv_fac_k2(e_feat2)
        d_feat2 = self.kconv_deblur(d_feat2, kernel_2)
        if self.connection:
            d_feat2 = d_feat2 + e_feat2

        d_feat1 = self.D_block1(d_feat2)  #64 1/2
        kernel_1  = self.conv_fac_k1(e_feat1)
        d_feat1 = self.kconv_deblur(d_feat1, kernel_1)
        if self.connection:
            d_feat1 = d_feat1 + e_feat1

        out = self.D_block0(d_feat1)

        if side_loss:
            return out_side, out
        else:
            return out
