import torch
from thop import profile
import numpy as np 

class FM_FCN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.m_attention_mask1 = None
        self.m_attention_mask2 = None
        self.m_attention_mask3 = None

        self.m_appearance_branch = AppearanceModel(in_channels=in_channels, out_channels=out_channels)
        self.m_motion_branch = MotionModel(in_channels=in_channels, out_channels=out_channels)
        self.m_filter_block = FilterBlock(out_channels * 2, 128)

    def forward(self, x):
        x = torch.chunk(x, 2, dim=1)
        self.m_attention_mask1, self.m_attention_mask2, self.m_attention_mask3 = self.m_appearance_branch(x[0].squeeze())
        motion_output = self.m_motion_branch(x[1].squeeze(), self.m_attention_mask1, self.m_attention_mask2, self.m_attention_mask3)
        out = self.m_filter_block(motion_output)
        return out

    def get_attention_mask(self):
        return self.m_attention_mask1, self.m_attention_mask2

class AppearanceModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.m_pcc_block = PCCBlock(in_channels, out_channels)
        self.m_attention_1 = AttentionBlock(out_channels)
        self.m_se_block = SEBlock(out_channels, out_channels * 2)
        self.m_attention_2 = AttentionBlock(out_channels * 2)
        self.m_sr_block = SRBlock(out_channels * 2)

    def forward(self, x):
        x = self.m_pcc_block(x)
        mask_1 = self.m_attention_1(x)
        x = self.m_se_block(x)
        mask_2 = self.m_attention_2(x)
        mask_3 = self.m_sr_block(x)
        return mask_1, mask_2, mask_3


class MotionModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.m_pcc_block = PCCBlock(in_channels, out_channels, True)
        self.m_se_block = SEBlock(out_channels, out_channels * 2, True)
        self.m_sr_block = SRBlock(out_channels * 2)

    def forward(self, x, mask_1, mask_2, mask_3):
        x = self.m_pcc_block(x, mask_1)
        x = self.m_se_block(x, mask_2)
        x = self.m_sr_block(x, mask_3)
        return x

class PCCBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_filter=False):
        super().__init__()
        self.m_conv_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3, 
                            padding=0),
            torch.nn.Tanh()
        )
        self.m_use_filter = use_filter
        if self.m_use_filter:
            self.m_filter_module = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 1, 1),
                                padding=(1, 0, 0)),
                torch.nn.Tanh()
            )
            
    def forward(self, x, mask=None):
        x = self.m_conv_module(x)
        if self.m_use_filter:
            x = x.transpose(0, 1)
            x = x.unsqueeze(0)
            x = self.m_filter_module(x)
            x = x.squeeze()
            x = x.transpose(0, 1)
        if mask is not None:
            x = x * mask
        return x
    
class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_filter=False):
        super().__init__()
        self.m_avg_pool = torch.nn.AvgPool2d(kernel_size=2)
        self.m_dropout = torch.nn.Dropout2d(p=0.5)
        self.m_conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3, 
                            padding=0),
            torch.nn.Tanh()
        )
        self.m_use_filter = use_filter
        if self.m_use_filter:
            self.m_filter_module = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 1, 1),
                                padding=(1, 0, 0)),
                torch.nn.Tanh()
            )
            
    def forward(self, x, mask=None):
        x = self.m_avg_pool(x)
        x = self.m_dropout(x)
        x = self.m_conv_block(x)
        if self.m_use_filter:
            x = x.transpose(0, 1)
            x = x.unsqueeze(0)
            x = self.m_filter_module(x)
            x = x.squeeze()
            x = x.transpose(0, 1)
        if mask is not None:
            x = x * mask
            
        return x
        
class SRBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.m_adapt_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.m_filter_module = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.Tanh(),
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.Tanh()
        )
        
    def forward(self, x, mask=None):
        x = self.m_adapt_avg_pool(x)
        x = x.squeeze().transpose(0, 1).unsqueeze(0)
        x = self.m_filter_module(x)
        if mask is not None:
            x = x + mask 
        return x
        

class FilterBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.m_filter_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=3,
                            padding=1),
            torch.nn.Tanh(),
        )
        self.m_filter_2 =  torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_channel,
                            out_channels=1,
                            kernel_size=3,
                            padding=1),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        x = self.m_filter_1(x)
        x = self.m_filter_2(x)
        x = x.transpose(0, 1)
        return x

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=1,  padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        return mask

if __name__ == '__main__':
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    img = torch.rand(256, 2, 3, 72, 72).to(device)  # [batch, norm + diff, channel, width, height]
    net = FM_FCN().to(device)
    
    y = net(img)
    criterion = torch.nn.L1Loss(reduction='mean')
    loss = criterion(y, y)
    loss.backward()
    
    # import time 
    # t1 = time.time()
    # for _ in range(100):
    #     out = net(img)  # [batch, 1]
    # t2= time.time()
    # print(t2-t1)

    # flops, params = profile(net, inputs=(img,))
    # print(flops/256)
    # print(params)