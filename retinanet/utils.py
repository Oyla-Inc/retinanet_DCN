import torch
import torch.nn as nn
import numpy as np
from dcn_v2 import dcn_v2_conv, DCNv2, DCN
from dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DeFormConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,bn=False,deformable_groups=1,only_dcn=False,use_depth=False):
        super(DeFormConvModule, self).__init__()

        #conv2d = DepthConv(inplanes,planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        channels_ = deformable_groups * 3 * kernel_size * kernel_size
        #print('inplanes',inplanes,channels_)
        self.use_depth = use_depth
        if self.use_depth:
            ip = 1
        else:
            ip = inplanes
        self.conv_offset_mask = nn.Conv2d(ip, channels_, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=True)
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
        
        conv2d = DCNv2(inplanes, planes, (kernel_size, kernel_size),
                       stride=stride, padding=padding, dilation=dilation,
                       deformable_groups=deformable_groups)
        
        layers = []
        if not only_dcn:
            if bn:
                layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
            else:
                layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*([conv2d]+layers))#(*layers)
        
    def forward(self, x, depth=None):
        #print('zxzz',x.shape,depth.shape)
        if self.use_depth:
            out = self.conv_offset_mask(depth)
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        #mask = torch.ones_like(mask)
        #print('xxx',offset.shape,x.shape,mask.shape)
        for im,module in enumerate(self.layers._modules.values()):
            if im==0:
                x = module(x,offset,mask)
            else:
                x = module(x)
        # x = self.conv2d(x, depth)
        # x = self.layers(x)
        return x

class BasicBlockWithDCN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_depth = False):
        super(BasicBlockWithDCN, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = DeFormConvModule(inplanes,planes,stride = stride, use_depth= use_depth, only_dcn=True, bn = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, depth = None):
        residual = x

        out = self.conv1(x,depth)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckWithDCN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_depth = False):
        super(BottleneckWithDCN, self).__init__()
        print("Doing Bottleneck With DCN")
        self.use_depth = use_depth
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv2 = DeFormConvModule(planes,planes,kernel_size = 3, padding = 1, stride = stride, use_depth= use_depth, only_dcn=True, bn = False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample is not None and use_depth:
            self.depth_downsampler =  nn.AvgPool2d(3,padding=1,stride=stride)
            
        self.stride = stride

    def forward(self, x):
        
        if self.use_depth:
            x,depth = x
            #assert x.shape[2:] == depth.shape[2:]
            
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, depth)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if depth is not None:
                depth = self.depth_downsampler(depth)
        out += residual
        out = self.relu(out)
        if self.use_depth:
            return out, depth
        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes
