from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms


class ISICDataset(Dataset):
    def __init__(self, datadir, csvpath, sketchdir, transform=None):
        self.datadir = datadir
        self.csv = pd.read_csv(csvpath)
        self.sketchdir = sketchdir
        self.transform = transform

    def __len__(self):
        return len(self.csv.index)

    def __getitem__(self, index):
        img_path = os.path.join(self.datadir, self.csv.iloc[index, 0] + ".jpg")
        image = Image.open(img_path)

        labels = self.csv.iloc[index, 1:].values
        label = np.argmax(labels, axis=0)

        label = torch.tensor(label)

        sketch_name = random.choice(os.listdir(self.sketchdir))
        sketch_path = os.path.join(self.sketchdir, sketch_name)
        fs, ext = os.path.splitext(sketch_path)

        while ext not in ['.jpg', '.jpeg']:
          sketch_name = random.choice(os.listdir(self.sketchdir))
          sketch_path = os.path.join(self.sketchdir, sketch_name)
          fs, ext = os.path.splitext(sketch_path)

        sketch = Image.open(sketch_path)

        if self.transform:
            image = self.transform(image)
            sketch = self.transform(sketch)

        return label, image, sketch  

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])




