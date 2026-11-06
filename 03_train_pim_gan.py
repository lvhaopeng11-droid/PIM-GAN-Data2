import os
import torch
from torch.utils.data import DataLoader
from pim_gan.data.dataset import MicrostructureDataset
from pim_gan.training.trainer import PIMGANTrainer

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(root, "data.csv")
    init_dir = os.path.join(root, "data/raw/in718_initial")
    final_dir = os.path.join(root, "data/raw/in718_final")
    
    print("Loading IN718 Training Dataset...")
    dataset = MicrostructureDataset(data_csv, init_dir, final_dir, split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    trainer = PIMGANTrainer(loader)
    print("Starting PIM-GAN Training (100 Epochs)...")
    for ep in range(1, 101):
        trainer.train_epoch(ep)

if __name__ == "__main__":
    main()
