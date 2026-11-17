"""Zero-DCE 네 손실. 공식 학습 코드의 가중치/정규화 convention을 사용."""
import torch
from torch import nn
from torch.nn import functional as F

class ZeroDCELoss(nn.Module):
    def forward(self, original, enhanced, curves):
        if min(original.shape[-2:]) < 16:
            raise ValueError('손실 계산에는 가로·세로 16픽셀 이상이 필요합니다.')
        means = enhanced.mean((2,3))
        differences = torch.stack((means[:,0]-means[:,1], means[:,0]-means[:,2], means[:,1]-means[:,2]),1)
        color = differences.square().norm(dim=1).mean()
        exposure = (F.avg_pool2d(enhanced.mean(1,keepdim=True),16)-0.6).square().mean()
        a = F.avg_pool2d(original.mean(1,keepdim=True),4)
        b = F.avg_pool2d(enhanced.mean(1,keepdim=True),4)
        kernels = original.new_zeros(4,1,3,3)
        kernels[:,0,1,1] = 1
        for i,(h,w) in enumerate(((1,0),(1,2),(0,1),(2,1))):
            kernels[i,0,h,w] = -1
        spatial = (F.conv2d(a,kernels,padding=1)-F.conv2d(b,kernels,padding=1)).square().sum(1).mean()
        # 공식 TV 구현은 채널 합을 보존합니다(채널 평균과 구별).
        tv = 2*((curves[:,:,1:]-curves[:,:,:-1]).square().sum(1).mean()
                +(curves[:,:,:,1:]-curves[:,:,:,:-1]).square().sum(1).mean())
        total = spatial+10*exposure+5*color+200*tv
        return total, {'spatial':spatial,'exposure':exposure,'color':color,'tv':tv}
