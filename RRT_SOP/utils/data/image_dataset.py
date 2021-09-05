from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, samples: list, transform):
        self.transform = transform

        self.categories = sorted(list(set([entry[1] for entry in samples])))
        self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        self.samples = [(path, self.cat_to_label[cat]) for path, cat in samples]
        self.targets = [label for _, label in self.samples]
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, index
