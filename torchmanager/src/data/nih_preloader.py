from typing import Optional, Callable

import os
from PIL import Image

import pandas as pd
import numpy as np

import torch
import pickle


class NIH_CXR_14_Preloader:
    metadata_file_name = "Data_Entry_2017.csv"
    
    classes = [
#         "No Finding",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
    ]
    
    train_list = "train_val_list.txt"
    test_list = "test_list.txt"
    
    def __init__(
            self,
            root: str,
            destination: str,
            train: bool = True,
            batch_size: int = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.destination = destination
        self.train = train
        self.batch_size = batch_size
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.process_metadata()
        
        self.save_dataset()
    
    def process_metadata(self) -> None:
        self.metadata = pd.read_csv(
            os.path.join(self.root, self.metadata_file_name),
            usecols=["Image Index", "Finding Labels"]
        ).reset_index().set_index("Image Index")
        
        self.metadata["Finding Labels"] = self.metadata["Finding Labels"].apply(lambda x: x.split("|"))
        self.metadata["One-Hot Labels"] = self.metadata["Finding Labels"].apply(lambda x: [1 if lbl in x else 0 for lbl in self.classes])
    
    def save_dataset(self):
        if self.train:
            data_list = self.train_list
            phase = "train"
        else:
            data_list = self.test_list
            phase = "test"
        
        if self.batch_size is None:
            self.batch_size = len(data_list)
        
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        main_dir = os.path.join(self.destination, phase)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
        
        samples = []
        
        with open(os.path.join(self.root, data_list), 'rt') as split_file:
            for line_idx, line in enumerate(split_file):
#                 print(f"{line_idx+1}", end='//')
                image_file_name = line.strip()
                
                image_file_path = self.get_image_path_from_name(image_file_name)
                image_labels = self.metadata.loc[image_file_name, "One-Hot Labels"]
                
                img, target = self.preprocess_sample(image_file_path, image_labels)
                
                sample = img, target
                samples.append(sample)
                
                if (line_idx+1) % self.batch_size == 0:
                    batch_num = line_idx//self.batch_size
                    print(f"saving batch {batch_num}...")
                    with open(os.path.join(main_dir, f"batch_{batch_num}.pkl"), 'wb') as file:
                        pickle.dump(samples, file)
                    samples = []
                    print(f"saved batch {batch_num}.")
                    
                    if batch_num == 1:
                        break
            
            if samples:
                print(f"saving residue batch...")
                batch_num = (len(split_file) - 1)//self.batch_size
                with open(os.path.join(main_dir, f"batch_{batch_num}.pkl"), 'wb') as file:
                    pickle.dump(samples, file)
                samples = []
                print(f"saved residue batch.")


    def get_image_path_from_name(self, image_file_name):
        image_file_index = self.metadata.loc[image_file_name, "index"]
        image_dir_num = int(1 + np.ceil((image_file_index - 4999 + 1)/10000))
        
        image_dir_name = "images_" + str(image_dir_num).zfill(3)
        image_path = os.path.join(self.root, image_dir_name, "images", image_file_name)
        
        return image_path
    
    def preprocess_sample(self, image_file_path, image_labels):
        img = self.get_pil_image(image_file_path)
        if self.transform is not None:
            img = self.transform(img)

        target = torch.FloatTensor(image_labels)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
        
    
    def get_pil_image(self, image_file_path):
        with open(image_file_path, "rb") as image_file:
            img = Image.open(image_file)
            return img.convert("RGB")
