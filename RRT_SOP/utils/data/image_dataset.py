from PIL import Image
import os.path as osp
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, samples, superpoints, max_desc, transform):
        self.transform = transform
        self.categories = sorted(list(set([entry[1] for entry in samples])))
        self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        self.samples = [(path, self.cat_to_label[cat]) for path, cat in samples]
        self.targets = [label for _, label in self.samples]
        self.superpoints = superpoints
        self.max_desc = max_desc
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        image_name = osp.splitext(osp.basename(image_path))[0]
        raw_points = self.superpoints[image_name]['keypoints']
        # raw_scores = self.superpoints[image_name]['scores']
        npts = min(len(raw_points), self.max_desc)
        points = torch.ones(self.max_desc, 2).float() * 0.5
        ptmask = torch.ones(self.max_desc, dtype=torch.bool)
        points[:npts] = torch.from_numpy(raw_points[:npts])
        ptmask[:npts] = False
        
        return image, points, ptmask, label, index
