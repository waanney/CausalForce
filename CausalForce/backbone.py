import torch
import torch.nn as nn
import timm
from torchvision.ops import roi_align


class ROI_ALIGN(nn.Module):
    def __init__(self, kernel_size, n, scale=1.0):
        """
            kernel_size: roi align kernel
            n: number of objects
        """
        super().__init__()
        self.roi_align = roi_align
        self.kernel = kernel_size
        self.scale = scale
        self.global_img = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.n = n

    def forward(self, features, boxes):
        b = len(boxes)
        boxes = list(boxes)

        x = self.roi_align(features, boxes, [self.kernel, self.kernel], self.scale)
        x = self.global_img(x)
        x = x.reshape(b, self.n, -1)
        
        return x


class Riskbench_backbone(nn.Module):
    def __init__(self, roi_align_kernel, n, pretrained=True, backbone='resnet50'):
        """
            backbone: specify which backbone ['resnet50', 'resnet101', ...]
        """
        super(Riskbench_backbone, self).__init__()
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, out_indices=[-1])
        self.object_layer = ROI_ALIGN(roi_align_kernel, n, 1./32.)


    def forward(self, img, bbox=None):
        """
            box : List[Tensor[N,4]] # box: (b t) n 4
        """
        object = None
        img = self.backbone(img)
        if bbox is not None:
            object = self.object_layer(img[0], bbox)  # (b t) N 2048 8 8

        return img, object
