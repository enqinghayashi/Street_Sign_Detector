import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import zipfile
import io

class ZipTestDataset(Dataset):
    def __init__(self, zip_path):
        self.zip_path = zip_path

        # Open zip once to get the list of files
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Filter only images
            self.file_list = [
                f for f in zf.namelist()
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                and not f.startswith('__MACOSX')
            ]

        # Sort to ensure consistent order
        self.file_list = sorted(self.file_list)

        # NEW: Match the training resolution (Height=64, Width=128)
        self.tf = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.zip_handle = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.zip_handle is None:
            self.zip_handle = zipfile.ZipFile(self.zip_path, 'r')

        file_path_in_zip = self.file_list[idx]

        # Read bytes
        img_bytes = self.zip_handle.read(file_path_in_zip)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply transforms
        img = self.tf(img)

        # clean filename
        clean_name = os.path.basename(file_path_in_zip)

        return img, clean_name
