""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
# from torchsummary import summary
from torch.cuda.amp import autocast

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm2d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out

class FusionLayer(nn.Module):
    def __init__(self, act):

        super(FusionLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = LUConv(1024, 512,act)
        self.layer2 = LUConv(512, 512,act)
        
    def forward(self, x1,x2):
        if x1.shape[2] != x2.shape[2]:
            m_batchsize, C, height, width = x1.size()
            x2 = F.interpolate(x1, size=(height, width), mode='trilinear',
                                     align_corners=True)
        concat = torch.cat((x1,x2),1)
        cov_layer1 = self.layer1(concat)
        cov_layer2 = self.layer2(cov_layer1)
        out = self.sigmoid(cov_layer2)
        return out

class MIA_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(MIA_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X z*y*x)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.out_channels = self.params['out_channels']
        self.bilinear = self.params['bilinear']
        self.factor = self.params['factor']
        self.dropout = self.params['dropout']
        feature_num = self.params['init_feature_num']

        self.inc = DoubleConv(self.in_channels, feature_num, dropout_p=self.dropout[0])
        self.down1 = DownBlock(feature_num, feature_num*2, self.dropout[1])
        self.down2 = DownBlock(feature_num*2, feature_num*4, self.dropout[2])
        self.down3 = DownBlock(feature_num*4, feature_num*8, self.dropout[3])
        self.down4 = DownBlock(feature_num*8, feature_num*16 // self.factor, self.dropout[4])
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]
    
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.out_channels = self.params['out_channels']
        self.bilinear = self.params['bilinear']
        self.factor = self.params['factor']
        feature_num = self.params['init_feature_num']

        self.up1 = UpBlock(feature_num*16, feature_num*8, self.bilinear, dropout_p=0.0)
        self.up2 = UpBlock(feature_num*8, feature_num*4, self.bilinear, dropout_p=0.0)
        self.up3 = UpBlock(feature_num*4, feature_num*2, self.bilinear, dropout_p=0.0)
        self.up4 = UpBlock(feature_num*2, feature_num * self.factor, self.bilinear, dropout_p=0.0)
        self.outc = OutConv(feature_num, self.out_channels)

    def forward(self, feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        x5 = feature[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class CMCNet(nn.Module):
    def __init__(self, in_channels, out_channels,init_feature_num=32, bilinear=False):
        super(CMCNet, self).__init__()
        factor = 2 if bilinear else 1
        params = {'in_channels': in_channels,
                  'init_feature_num': init_feature_num,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                #   'dropout': [0.1, 0.0, 0.0, 0.0, 0.0],
                  'out_channels': out_channels,
                  'bilinear': bilinear,
                  'factor': factor}
        self.encoder_1 = Encoder(params)
        self.encoder_2 = Encoder(params)
        self.decoder_1 = Decoder(params)
        self.decoder_2 = Decoder(params)

        self.MIA_module_mode_1 = MIA_Module()
        self.MIA_module_mode_2 = MIA_Module()
        
        self.fusion_layer = FusionLayer(act='relu')
        self.conv2d_convert = nn.Sequential(
            nn.GroupNorm(16, 1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )
    def forward_CMC(self, img_mode_1, img_mode_2):
        encoder_feat_mode_1 = self.encoder_1(img_mode_1)
        encoder_feat_mode_2 = self.encoder_2(img_mode_2)

        feat_for_CMC_mode_1 = encoder_feat_mode_1[-1]
        feat_for_CMC_mode_2 = encoder_feat_mode_2[-1]
        
        
        mode_1_F_mia = self.MIA_module_mode_1(feat_for_CMC_mode_1)
        mode_2_F_mia = self.MIA_module_mode_2(feat_for_CMC_mode_2)

        out_fuse = self.fusion_layer(mode_1_F_mia, mode_2_F_mia)
        
        mode_1_F_z = torch.cat([out_fuse, mode_1_F_mia], dim=1)
        mode_2_F_z = torch.cat([out_fuse, mode_2_F_mia], dim=1)
        
        mode_1_F_z = self.conv2d_convert(mode_1_F_z)
        mode_2_F_z = self.conv2d_convert(mode_2_F_z)

        encoder_feature_after_CMC_mode_1 = encoder_feat_mode_1[:4] + [mode_1_F_z]
        encoder_feature_after_CMC_mode_2 = encoder_feat_mode_2[:4] + [mode_2_F_z]
            
        mode_1_mask = self.decoder_1(encoder_feature_after_CMC_mode_1)
        mode_2_mask = self.decoder_2(encoder_feature_after_CMC_mode_2)

        return feat_for_CMC_mode_1, feat_for_CMC_mode_2, mode_1_mask, mode_2_mask

    def forward_onlyCSC(self, img_mode_1, img_mode_2): 
        encoder_feat_mode_1 = self.encoder_1(img_mode_1)
        encoder_feat_mode_2 = self.encoder_2(img_mode_2)
        
        feat_for_CMC_mode_1 = encoder_feat_mode_1[-1]
        feat_for_CMC_mode_2 = encoder_feat_mode_2[-1]

        output_mode_1 = self.decoder_1(encoder_feat_mode_1)
        output_mode_2 = self.decoder_2(encoder_feat_mode_2)
        
        return feat_for_CMC_mode_1, feat_for_CMC_mode_2, output_mode_1, output_mode_2
    
    def forward_onlyCSC_with_sigmoid(self, img_mode_1, img_mode_2): 
        encoder_feat_mode_1 = self.encoder_1(img_mode_1)
        encoder_feat_mode_2 = self.encoder_2(img_mode_2)
        
        feat_for_CMC_mode_1 = encoder_feat_mode_1[-1]
        feat_for_CMC_mode_2 = encoder_feat_mode_2[-1]

        output_mode_1 = self.decoder_1(encoder_feat_mode_1)
        output_mode_2 = self.decoder_2(encoder_feat_mode_2)
        
        return feat_for_CMC_mode_1, feat_for_CMC_mode_2, torch.sigmoid(output_mode_1), torch.sigmoid(output_mode_2)
    
    def forward_CMC_with_sigmoid(self, img_mode_1, img_mode_2):
        encoder_feat_mode_1 = self.encoder_1(img_mode_1)
        encoder_feat_mode_2 = self.encoder_2(img_mode_2)

        feat_for_CMC_mode_1 = encoder_feat_mode_1[-1]
        feat_for_CMC_mode_2 = encoder_feat_mode_2[-1]
        
        
        mode_1_F_mia = self.MIA_module_mode_1(feat_for_CMC_mode_1)
        mode_2_F_mia = self.MIA_module_mode_2(feat_for_CMC_mode_2)

        out_fuse = self.fusion_layer(mode_1_F_mia, mode_2_F_mia)
        
        mode_1_F_z = torch.cat([out_fuse, mode_1_F_mia], dim=1)
        mode_2_F_z = torch.cat([out_fuse, mode_2_F_mia], dim=1)
        
        mode_1_F_z = self.conv2d_convert(mode_1_F_z)
        mode_2_F_z = self.conv2d_convert(mode_2_F_z)

        encoder_feature_after_CMC_mode_1 = encoder_feat_mode_1[:4] + [mode_1_F_z]
        encoder_feature_after_CMC_mode_2 = encoder_feat_mode_2[:4] + [mode_2_F_z]
            
        mode_1_mask = self.decoder_1(encoder_feature_after_CMC_mode_1)
        mode_2_mask = self.decoder_2(encoder_feature_after_CMC_mode_2)

        return feat_for_CMC_mode_1, feat_for_CMC_mode_2, torch.sigmoid(mode_1_mask), torch.sigmoid(mode_2_mask)
    
    def forward_onlyMIA(self, img_mode_1, img_mode_2):
        return self.forward_CMC(img_mode_1, img_mode_2)
    
    def forward_onlyMIA_with_sigmoid(self, img_mode_1, img_mode_2):
        return self.forward_CMC_with_sigmoid(img_mode_1, img_mode_2)
    
    
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


if __name__ == '__main__':
    device = torch.device('cpu')  # cuda:0
    # inputs = torch.rand(2, 3, 144, 144).to(device)
    # net = UNet(in_channels=1, out_channels=1,init_feature_num=64)
    # res = net(inputs)
    # print('res shape:', res.shape)
    
    # feat = net.encoder_forward(inputs)
    # print(feat[1,:,:].shape)
    cmc = CMCNet(1,1,32)
    input_mode_1 = torch.rand(2,1,144,144)
    input_mode_2 = torch.rand(2,1,144,144)
    mode_1_F_ds, mode_2_F_ds, mode_1_mask, mode_2_mask = cmc(input_mode_1, input_mode_2)
    # print(input_mode_1)
    # print(mode_1_F_ds)
    print(mode_2_F_ds.shape)
    print(mode_1_mask.shape)
    print(mode_2_mask.shape)
    
    
    
    

    