import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/workspace/data"))

import sys

from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim

import torchvision.transforms as transforms
import torchvision

# from fastprogress import master_bar, progress_bar

from PIL import Image
import pickle

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data import Dataset
from PIL import Image

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


train_data = pd.read_csv("/workspace/data/VinDr-CXR/annotations/image_labels_train.csv")
test_data = pd.read_csv("/workspace/data/VinDr-CXR/annotations/image_labels_test.csv")
print(train_data.head(5))
print(test_data.head(5))
LABELS = train_data.columns[2:]
print(LABELS)

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=2019)

IMAGE_SIZE = 224                              # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
BATCH_SIZE = 64 

class ChestXrayDataset(Dataset):
    
    def __init__(self, folder_dir, dataframe, image_size, label_index,normalization):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        self.image_paths = [] # List of image paths
        self.image_labels = [] # List of image labels
        
        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()

        ]
        
        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        
        self.image_transformation = transforms.Compose(image_transformation)
        
        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row[0] + ".dicom")
            self.image_paths.append(image_path)
            if len(row) < 29:
                labels = [0] * 28
            else:
                labels = []
                for col in row[label_index:]:
                    if col == 1:
                        labels.append(1)
                    else:
                        labels.append(0)
            self.image_labels.append(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.image_paths[index]
        image_data = Image.fromarray(read_xray(image_path))
        
        # TODO: Image augmentation code would be placed here
        
        # Resize and convert image to torch tensor 
        image_data = self.image_transformation(image_data)
        
        return image_data, torch.FloatTensor(self.image_labels[index])

train_dataset = ChestXrayDataset("/workspace/data/VinDr-CXR/train", train_data, IMAGE_SIZE, 2, False)

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
# for data, label in train_dataloader:
#     print(data.size())
#     print(label.size())
#     break
samples = []
for ind,img in enumerate(train_dataset):
    samples.append(img)
    # if(ind==100):
    #     break

with open(os.path.join("/workspace/data/Processed_VinBig_test/train", f"batch_0.pkl"), 'wb') as file:
    pickle.dump(samples, file)
        # break
test_dataset = ChestXrayDataset("/workspace/data/VinDr-CXR/test", test_data, IMAGE_SIZE, 1, False)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
samples = []
for ind,img in enumerate(test_dataset):
    samples.append(img)
    # if(ind==100):
    #     break
# for ind,batch in enumerate(test_dataloader):
with open(os.path.join("/workspace/data/Processed_VinBig_test/test", f"batch_0.pkl"), 'wb') as file:
    pickle.dump(samples, file)

print("----DONE----")