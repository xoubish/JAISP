# train_stage1_jepa_lambda.py
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler #

from JAISP_dataset import make_loader
from stage1_jepa_lambda_foundation import JAISPFoundation, create_optimizer, create_scheduler

class Stage1LambdaTrainer:
    def __init__(self, rubin_dir, euclid_dir, output_dir="./checkpoints", batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.dataset, self.dataloader = make_loader(
            rubin_dir=rubin_dir, euclid_dir=euclid_dir,
            batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.model = JAISPFoundation().to(self.device)

    def log_pca(self, outputs, step):
        z_r = outputs['z_rubin'][0].detach().cpu().numpy()
        H, W, D = z_r.shape
        pca = PCA(n_components=3)
        rgb = pca.fit_transform(z_r.reshape(-1, D)).reshape(H, W, 3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        wandb.log({"vis/pca_map": wandb.Image(rgb)}, step=step)

    def train(self, epochs=100, lr=1e-4):
        optimizer = create_optimizer(self.model, lr=lr)
        scheduler = create_scheduler(optimizer, 20, epochs)
        scaler = GradScaler() #
        
        wandb.init(project="JAISP-Lambda-Foundation", name="Dense-Resolution-Run")

        global_step = 0
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                optimizer.zero_grad()
                
                # Mixed Precision Forward
                with autocast():
                    outputs = self.model(batch)
                    loss = outputs['loss']
                
                # Scaled Backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                if global_step % 10 == 0:
                    wandb.log({"train/loss": loss.item()}, step=global_step)
                if global_step % 100 == 0:
                    self.log_pca(outputs, global_step)
                global_step += 1
            scheduler.step()

def main():
    trainer = Stage1LambdaTrainer(
        rubin_dir="../data/rubin_tiles_ecdfs",
        euclid_dir="../data/euclid_tiles_ecdfs",
        batch_size=4 # Start small to verify memory stability
    )
    trainer.train()

if __name__ == "__main__":
    main()