import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .crop_tiles import extract_regular_crops, resize_crop_to_network_dim
from .augment import augment_crop_twofold
from .intensity import normalize_zero_one_to_tanh

class MicrostructureDataset(Dataset):
    def __init__(self, data_csv_path, initial_dir, final_dir, split="train", transform=True):
        self.df = pd.read_csv(data_csv_path)
        self.initial_dir = initial_dir
        self.final_dir = final_dir
        self.transform = transform
        self.split = split
        
        self.records = []
        for idx, row in self.df.iterrows():
            t1 = float(row["Temperature"])
            t2 = float(row.get("Temp2", 0.0))
            if t1 == 980.0 and t2 == 0.0:
                rec_split = "level2_980_holdout"
            elif idx % 7 == 0:
                rec_split = "validation"
            elif idx % 50 == 0:
                rec_split = "level1_internal_test"
            else:
                rec_split = "train"
                
            if rec_split == self.split:
                init_id = (idx % 6) + 1
                self.records.append({
                    "id": idx,
                    "init_path": os.path.join(initial_dir, f"initial_{init_id}.tif"),
                    "final_path": os.path.join(final_dir, f"final_{idx}.tif"),
                    "T": t1,
                    "t": float(row["Time"])
                })

    def __len__(self):
        return len(self.records) * 12 * (2 if self.transform else 1)

    def __getitem__(self, idx):
        parent_idx = idx // 24 if self.transform else idx // 12
        sub_idx = idx % 24 if self.transform else idx % 12
        crop_id = sub_idx % 12
        is_aug = (sub_idx >= 12)
        
        item = self.records[parent_idx]
        init_img = self._load_img(item["init_path"])
        final_img = self._load_img(item["final_path"])
        
        row, col = crop_id // 4, crop_id % 4
        x0, y0 = col * 425, row * 425
        
        c_init = init_img[y0:y0+425, x0:x0+425]
        c_final = final_img[y0:y0+425, x0:x0+425]
        
        c_init = resize_crop_to_network_dim(c_init)
        c_final = resize_crop_to_network_dim(c_final)
        
        if is_aug:
            c_init = np.fliplr(c_init)
            c_final = np.fliplr(c_final)
            
        t_init = torch.from_numpy(normalize_zero_one_to_tanh(c_init / 255.0)).unsqueeze(0).float()
        t_final = torch.from_numpy(normalize_zero_one_to_tanh(c_final / 255.0)).unsqueeze(0).float()
        
        return {
            "initial": t_init,
            "final": t_final,
            "T": torch.tensor(item["T"], dtype=torch.float32),
            "t": torch.tensor(item["t"], dtype=torch.float32)
        }

    def _load_img(self, path):
        if os.path.exists(path):
            return np.array(Image.open(path).convert("L"))
        return np.ones((1275, 1920), dtype=np.uint8) * 128
