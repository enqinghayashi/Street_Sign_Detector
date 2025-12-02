import os
import json
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]



# Dataset extraction into single digit samples
class SingleDigitCropDataset(Dataset):
    def __init__(self, json_path, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.samples = [] 
        
        # Load JSON and flatten it into individual digit samples
        # Format: (filename, box_params, label_value)
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        print("Parsing JSON to create crop list...")
        for filename, data in meta.items():
            # data contains: 'height', 'label', 'left', 'top', 'width' lists
            labels = data['label']
            tops = data['top']
            lefts = data['left']
            heights = data['height']
            widths = data['width']
            
            for i in range(len(labels)):
                # Create a sample for every single digit
                label = int(labels[i])
                
                # Fix label 10 -> 0 if necessary 
                # Tianchi usually uses 0-9 directly, but just in-case
                if label == 10: label = 0
                
                box = (lefts[i], tops[i], widths[i], heights[i])
                self.samples.append((filename, box, label))
                
        print(f"Found {len(self.samples)} individual digits.")
        self.zip_handle = None

    def __len__(self):
        return len(self.samples)




# Trainin logic


if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    train_reader()