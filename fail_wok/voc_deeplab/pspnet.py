import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import models
# import resnet as models
from torchE.nn import SyncBatchNorm2d

# from utils import init_weights


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm=nn.BatchNorm2d):
        super(PPM, self).__init__()
        self.features = []
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=3, dilation =3, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=6, dilation =6, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=9, dilation =9, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(f(x))
            #out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

class PSPNet(nn.Module):
    def __init__(self, backbone='resnet', layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, use_softmax=False, use_aux=False, pretrained=True, syncbn=True, group_size=8, group=None):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.use_softmax = use_softmax
        self.use_aux = use_aux

        if backbone == 'resnet':
            import resnet as models
        elif backbone == 'ibnnet_a':
            import ibnnet_a as models
        elif backbone == 'ibnnet_b':
            import ibnnet_b as models
        else:
            raise NameError('Backbone type not defined!')

        if syncbn:
            # from lib.syncbn import SynchronizedBatchNorm2d as BatchNorm
            def BNFunc(*args, **kwargs):
                return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=True)
            BatchNorm = BNFunc
        else:
            from torch.nn import BatchNorm2d as BatchNorm
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        if backbone == 'ibnnet_b':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.INCat0, resnet.relu, resnet.maxpool)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n and not 'convbnin.conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n and not 'convbnin.conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            
        self.cls = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.result = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(256, 20, 1, bias=False))
        if use_aux:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            # init_weights(self.aux)
        # comment to use default initialization
        # init_weights(self.ppm)
        # init_weights(self.cls)

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        if self.use_aux:
            aux = self.aux(x)
            if self.zoom_factor != 1:
                aux = F.upsample(aux, size=(h, w), mode='bilinear')
        x = self.layer4(x)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        x = self.result(x)



        if self.use_softmax:
            return F.log_softmax(x, dim = 1)
        else:
            return x

            
    def forward_cam(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        if self.use_aux:
            aux = self.aux(x)
            if self.zoom_factor != 1:
                aux = F.upsample(aux, size=(h, w), mode='bilinear')
        x = self.layer4(x)
        if self.use_ppm:
            x = self.ppm(x)[:,-256*2:-256,:,:].contiguous()
        x = F.conv2d(x, self.cls[0].weight[:,-256*2:-256:,:].contiguous())
        x = self.cls[1](x)
        x = self.cls[2](x)
        x = self.cls[3](x)
        #x = self.cls(x)
        x = F.conv2d(x, self.result[1].weight)#[:27,:,:,:]
        x = F.relu(x)
        #print(self.result.weight)



        if self.use_softmax:
            if self.use_aux:
                aux = F.log_softmax(aux, dim=1)
                return x, aux
            else:
                return x
        else:
            if self.use_aux:
                return x, aux
            else:
                return x            
                 
            
            

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sim_data = torch.autograd.Variable(torch.rand(2, 3, 473, 473)).cuda(async=True)
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, use_softmax=True, use_aux=True, pretrained=True, syncbn=True).cuda()
    print(model)
    output, _ = model(sim_data)
    print('PSPNet', output.size())
