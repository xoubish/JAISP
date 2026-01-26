import torch
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from JAISP_dataset import make_loader
from stage1_superpoint_jepa import SuperPointJEPA, create_optimizer, create_scheduler

class SuperPointJEPATrainer:
    def __init__(self, rubin_dir, euclid_dir, batch_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset, self.dataloader = make_loader(rubin_dir, euclid_dir, batch_size=batch_size, shuffle=True)
        self.model = SuperPointJEPA().to(self.device)

    def train(self, epochs=100, lr=5e-5):
        opt = create_optimizer(self.model, lr); sched = create_scheduler(opt, 20, epochs)
        scaler = GradScaler('cuda')
        
        # Force W&B to initialize and define metrics explicitly
        wandb.init(project="JAISP-SuperPoint-JEPA")
        wandb.define_metric("train/*", step_metric="global_step")
        
        global_step = 1
        for ep in range(epochs):
            self.model.train()
            pbar = tqdm(self.dataloader, desc=f"Ep {ep+1}")
            for batch in pbar:
                opt.zero_grad()
                with autocast('cuda'):
                    out = self.model(batch)
                    loss = out['loss']
                
                if not torch.isfinite(loss): continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                scaler.step(opt); scaler.update()
                
                # Explicitly log to the defined metrics
                wandb.log({
                    'train/loss': loss.item(), 
                    'train/similarity': out['similarity'],
                    'global_step': global_step
                })
                global_step += 1
            
            # Periodically force a sync of media/images
            self.model.eval()
            with torch.no_grad():
                v_out = self.model(batch)
            self.visualize_matching(batch, v_out, global_step)
            sched.step()

    def visualize_matching(self, batch, outputs, step):
        def to_np(k): return torch.stack(batch[k])[0].cpu().numpy() if isinstance(batch[k], list) else batch[k][0].cpu().numpy()
        r_rgb = np.clip(np.stack([to_np('x_rubin')[3], to_np('x_rubin')[2], to_np('x_rubin')[1]], -1), 0, 1)
        e_rgb = np.clip(np.stack([to_np('x_euclid')[2], to_np('x_euclid')[1], to_np('x_euclid')[0]], -1), 0, 1)
        
        max_h = max(r_rgb.shape[0], e_rgb.shape[0])
        r_pad = np.pad(r_rgb, ((0, max_h-r_rgb.shape[0]), (0, 0), (0, 0)))
        e_pad = np.pad(e_rgb, ((0, max_h-e_rgb.shape[0]), (0, 0), (0, 0)))
        
        combined = np.hstack([r_pad, e_pad])
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(combined, origin='lower')
        matches = outputs['matches'][0].cpu().numpy()
        w1 = r_rgb.shape[1]
        for i in range(0, len(matches), 50):
            ry, rx = (i // 32) * 16, (i % 32) * 16
            ey, ex = (matches[i] // 65) * 16, (matches[i] % 65) * 16 + w1
            ax.plot([rx, ex], [ry, ey], color='lime', alpha=0.4)
        
        plt.axis('off')
        wandb.log({
            "val/alignment_lines": wandb.Image(fig),
            "val/rubin_heatmap": wandb.Image(outputs['rubin_heatmap'][0,0].cpu().numpy()),
            "val/euclid_heatmap": wandb.Image(outputs['euclid_heatmap'][0,0].cpu().numpy()),
            "global_step": step
        })
        plt.close(fig)

def main():
    trainer = SuperPointJEPATrainer(
        rubin_dir="../data/rubin_tiles_ecdfs", 
        euclid_dir="../data/euclid_tiles_ecdfs", 
        batch_size=1
    )
    trainer.train(epochs=100, lr=5e-5)

if __name__ == "__main__":
    main()