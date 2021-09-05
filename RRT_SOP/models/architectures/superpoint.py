from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    # oh, ow = h*s, w*s
    # keypoints[:,:,0] *= ow
    # keypoints[:,:,1] *= oh
    # keypoints = keypoints - s/2 + 0.5
    # keypoints[:,0] /= (w*s - s/2 - 0.5)
    # keypoints[:,1] /= (h*s - s/2 - 0.5)
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = F.grid_sample(descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPointNet(BaseModel):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    def __init__(self, freeze_backbone, **kwargs):
        super(SuperPointNet, self).__init__(**kwargs)
        self.backbone_features = 256
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
        path = Path(__file__).parent / 'superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path), map_location=torch.device('cpu')), strict=False)
        print('Loaded SuperPoint model')

        ############################################################
        if freeze_backbone:
            for param in self.conv1a.parameters():
                param.requires_grad = False
            for param in self.conv1b.parameters():
                param.requires_grad = False
            for param in self.conv2a.parameters():
                param.requires_grad = False
            for param in self.conv2b.parameters():
                param.requires_grad = False
            for param in self.conv3a.parameters():
                param.requires_grad = False
            for param in self.conv3b.parameters():
                param.requires_grad = False
            for param in self.conv4a.parameters():
                param.requires_grad = False
            for param in self.conv4b.parameters():
                param.requires_grad = False
            for param in self.convPa.parameters():
                param.requires_grad = False
            for param in self.convPb.parameters():
                param.requires_grad = False
            for param in self.convDa.parameters():
                param.requires_grad = False
            for param in self.convDb.parameters():
                param.requires_grad = False

    def feature_extractor(self, images, points):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder
        x = self.relu(self.conv1a(images))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        # print('descriptors',descriptors.shape)

        g = F.adaptive_avg_pool2d(descriptors, (1,1))
        g = torch.flatten(g, 1)

        # Extract descriptors
        l = sample_descriptors(points, descriptors, 8)

        return g, torch.transpose(l, 1, 2)


def superpoint(freeze_backbone, **kwargs) -> nn.Module:
    model = SuperPointNet(freeze_backbone, **kwargs)
    return model