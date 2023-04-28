import torch
import torchvision.transforms as transforms

class DatasetWithAttribute(torch.utils.data.Dataset):
    def __init__(self, parent_dataset, attribute_list):
        self.parent_dataset = parent_dataset
        self.attribute_list = attribute_list

        self.weight_index_present = len(parent_dataset[0]) > 2

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, idx):
        return (self.parent_dataset[idx][0],
                self.parent_dataset[idx][1],
                self.parent_dataset[idx][2] if self.weight_index_present else 1,
                self.attribute_list[idx])

class ClasswiseTransformDataset(torch.utils.data.Dataset):
    def __init__(self, parent_dataset, in_transforms, out_transforms, classes, p=1):
        self.parent_dataset = parent_dataset
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms
        self.classes = set(classes)
        self.p = p

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, idx):
        flip = torch.rand(1)[0] > self.p

        label = self.parent_dataset[idx][1]
        if (label in self.classes) != flip: # XOR
            return (self.in_transforms(self.parent_dataset[idx][0]), *self.parent_dataset[idx][1:])
        return (self.out_transforms(self.parent_dataset[idx][0]), *self.parent_dataset[idx][1:])

class RandomSquareTransform:
    def __init__(self, width, img_width):
        square = torch.zeros(3, width, width)
        square[0:2] = 1
        self.square = square.view(-1)

        region = torch.zeros(3, img_width, img_width, dtype=bool)
        region[:, :width, :width] = 1
        roll_dist = (img_width - width) // 2
        self.region = region.roll((roll_dist, roll_dist), dims=(1,2))

        translation_dist = (img_width - width) / img_width / 2
        self.translate = transforms.RandomAffine(0, translate=(translation_dist, translation_dist))

    def __call__(self, x):
        x = x.clone()

        region = self.translate(self.region)
        x[region] = self.square

        return x
    
