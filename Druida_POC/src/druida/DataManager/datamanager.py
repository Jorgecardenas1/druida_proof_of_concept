import os
from ..setup import inputType

import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#from torchvision import datasets
#from torchvision.io import read_image


class VectorSet(Dataset):
    def __init__(self, pandas_df, type='float'):
        super(VectorSet,self).__init__()

        if type=='float':
            print(type)
            self.vector = torch.tensor(pandas_df.values).type(torch.FloatTensor)
        else:
            self.vector = torch.tensor(pandas_df.values)


    def __len__(self):
        return self.vector.shape[0]

    def __getitem__(self, index):
        return self.vector[index]
    
        #If we want to use different types of entries as vector
    def modeling(self, object):
            
            self.input_type = object['type']
            self.inputData=self.vector

            if inputType['image'] == self.input_type:
                flatten = nn.Flatten()
                self.flat_image = flatten(self.inputData)
                return self.flat_image
            
            elif inputType['vector'] == self.input_type:
                
                #any shaping or forming of our tensor
                """ cuda0 = torch.device('cuda:0')
                tensor.to(cuda0)
                

                tensor.to(cuda0, dtype=torch.float64)
                
                other = torch.randn((), dtype=torch.float64, device=cuda0)
                tensor.to(other, non_blocking=True) """

                
                device = torch.device(object['device'])
                self.tensor = self.vector.to(device, dtype=object['torchType'])
                return self.tensor
            else:
                pass




class ImageSet(Dataset):
    def __init__(self, annotations_file, 
                 img_dir, transform=None, 
                 target_transform=None):
        
        super(ImageSet,self).__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform =  transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    # loads and returns a sample from the dataset at the given index idx

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def load(self):
        pass

    def create(self):
        print(str(self.name))

    

