import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Fitzpatrick17k(Dataset): 
    """A PyTorch dataset class for the Fitzpatrick17k skin lesion dataset"""

    def __init__(self, root, fitz_values=[1,2,3,4,5,6], transform=None):
        self.root = root
        self.transform = transforms.ToTensor() if transform is None else transform

        # Build an index dataframe with file locations
        self.index_df = pd.read_csv(os.path.join(root, 'fitzpatrick17k/fitzpatrick17k.csv'))
        found_images = [image_name.replace('.jpg', '')
                        for image_name in os.listdir(os.path.join(root, 'fitzpatrick17k/images'))]
        self.index_df = self.index_df.loc[self.index_df.md5hash.isin(found_images)]
        
        self.index_df = self.index_df.loc[lambda x: x.fitzpatrick.isin(fitz_values)]

        self.index_df['filepath'] = self.index_df.md5hash \
            .apply(lambda x: os.path.join(root, 'fitzpatrick17k/images', x + '.jpg'))
        
        # Build a label key to translate labels to integers
        all_labels = self.index_df.three_partition_label.drop_duplicates().sort_values().reset_index(drop=True)
        self.label_key = pd.Series(data=all_labels.index, index=all_labels.values).to_dict() 
        
        self.nclasses = len(self.label_key)
        
    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):  
        sample = self.index_df.iloc[idx]

        label = self.label_key[sample.three_partition_label]

        img = Image.open(sample.filepath)
        img = self.transform(img)

        return img, label
