import numpy as np
import bisect, torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .delf import datum_io
from .delf import feature_io


class FeatureDataset(Dataset):
    def __init__(self, 
            data_dir: str,
            samples: list,
            desc_name: str, 
            max_sequence_len: int,
            gnd_data=None):
        self.data_dir = data_dir
        self.desc_name = desc_name
        self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
        self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        self.samples = [(entry[0], self.cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
        self.targets = [entry[1] for entry in self.samples]
        self.gnd_data = gnd_data
        self.max_sequence_len = max_sequence_len
        self.scales = [0.5, 0.70710677, 1., 1.41421354, 2., 2.82842708, 4.]
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, load_image=False):
        '''
        Output
            global_desc: (2048, )
            local_desc: (max_sequence_len, 128)
            local_mask: (max_sequence_len, )
            scale_inds: (max_sequence_len, )
            positions: (max_sequence_len, 2)
            label: int
            name: str
        '''
        image_path, label, width, height = self.samples[index]
        image_name  = osp.splitext(osp.basename(image_path))[0]
        global_path = osp.join(self.data_dir, 'delg_%s'%self.desc_name, image_name+'.delg_global')
        local_path  = osp.join(self.data_dir, 'delg_%s'%self.desc_name, image_name+'.delg_local')
        assert(osp.exists(global_path) and osp.exists(local_path))
        global_desc = datum_io.ReadFromFile(global_path)
        locations, scales, desc, attention, _ = feature_io.ReadFromFile(local_path)

        local_mask = torch.ones(self.max_sequence_len, dtype=torch.bool)
        local_desc = np.zeros((self.max_sequence_len, 128), dtype=np.float32)
        scale_inds = torch.zeros(self.max_sequence_len).long()
        seq_len = min(desc.shape[0], self.max_sequence_len)
        local_desc[:seq_len] = desc[:seq_len]
        local_mask[:seq_len] = False
        scale_inds[:seq_len] = torch.as_tensor([bisect.bisect_right(self.scales, s) for s in scales[:seq_len]]).long() - 1

        ###############################################
        # Sine embedding
        positions = torch.zeros(self.max_sequence_len, 2).float()
        normx = locations[:, 1]/float(width)
        normy = locations[:, 0]/float(height)
        positions[:seq_len] = torch.from_numpy(np.stack([normx, normy], -1)).float()[:seq_len]
        ##############################################
        
        global_desc = torch.from_numpy(global_desc).float()
        local_desc = torch.from_numpy(local_desc).float()
        
        if load_image:
            image = Image.open(osp.join(self.data_dir, image_path)).convert('RGB')
            image = image.resize((512, 512))
            return F.to_tensor(image), global_desc, local_desc, local_mask, scale_inds, positions, label, image_name
        else:
            return global_desc, local_desc, local_mask, scale_inds, positions, label, image_name