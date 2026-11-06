from .seed import set_seed
from .trainer import PIMGANTrainer

def run_multiseed_training(dataloader, seeds=[42, 43, 44, 45, 46], epochs=100):
    for s in seeds:
        print(f"=== Starting Training with Random Seed: {s} ===")
        set_seed(s)
        trainer = PIMGANTrainer(dataloader)
        for ep in range(1, epochs + 1):
            trainer.train_epoch(ep)
