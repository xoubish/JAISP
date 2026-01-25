# monitor_training.py
#
# Live training monitor for Stage 1
# Periodically checks checkpoints and updates visualizations

import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime


class TrainingMonitor:
    """Monitor training progress in real-time"""
    
    def __init__(self, checkpoint_dir, output_dir="./monitoring", update_interval=60):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.update_interval = update_interval  # seconds
        
        self.history = {
            'epochs': [],
            'losses': [],
            'timestamps': []
        }
    
    def check_for_updates(self):
        """Check for new checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt['epoch']
            
            # Only add if new
            if epoch not in self.history['epochs']:
                self.history['epochs'].append(epoch)
                self.history['losses'].append(ckpt['loss'])
                self.history['timestamps'].append(datetime.now())
                print(f"[{datetime.now().strftime('%H:%M:%S')}] New checkpoint: Epoch {epoch}, Loss: {ckpt['loss']:.4f}")
    
    def plot_live_curves(self):
        """Update training curve plots"""
        if len(self.history['epochs']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curve
        ax = axes[0, 0]
        ax.plot(self.history['epochs'], self.history['losses'], 'o-', 
               linewidth=2, markersize=6, color='#3498DB')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)
        
        # Best loss
        if len(self.history['losses']) > 0:
            best_idx = np.argmin(self.history['losses'])
            ax.plot(self.history['epochs'][best_idx], self.history['losses'][best_idx], 
                   'r*', markersize=15, label=f'Best: {self.history["losses"][best_idx]:.4f}')
            ax.legend()
        
        # Loss improvement rate
        ax = axes[0, 1]
        if len(self.history['losses']) > 1:
            improvements = -np.diff(self.history['losses'])
            ax.plot(self.history['epochs'][1:], improvements, 'o-', 
                   linewidth=2, markersize=6, color='#2ECC71')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss Improvement', fontsize=11)
            ax.set_title('Loss Improvement per Epoch', fontsize=12, weight='bold')
            ax.grid(alpha=0.3)
        
        # Training speed
        ax = axes[1, 0]
        if len(self.history['timestamps']) > 1:
            time_diffs = [(self.history['timestamps'][i] - self.history['timestamps'][i-1]).total_seconds() / 60
                         for i in range(1, len(self.history['timestamps']))]
            ax.plot(self.history['epochs'][1:], time_diffs, 'o-', 
                   linewidth=2, markersize=6, color='#F39C12')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Time per Epoch (minutes)', fontsize=11)
            ax.set_title('Training Speed', fontsize=12, weight='bold')
            ax.grid(alpha=0.3)
        
        # Progress summary
        ax = axes[1, 1]
        ax.axis('off')
        
        if len(self.history['epochs']) > 0:
            current_epoch = self.history['epochs'][-1]
            current_loss = self.history['losses'][-1]
            best_loss = min(self.history['losses'])
            
            summary_text = f"""
            Training Summary
            ═══════════════════════════
            
            Current Epoch:     {current_epoch}
            Current Loss:      {current_loss:.4f}
            Best Loss:         {best_loss:.4f}
            
            Total Checkpoints: {len(self.history['epochs'])}
            
            """
            
            if len(self.history['timestamps']) > 1:
                elapsed = (self.history['timestamps'][-1] - self.history['timestamps'][0]).total_seconds() / 60
                avg_time = elapsed / (len(self.history['timestamps']) - 1)
                summary_text += f"Avg Time/Epoch:    {avg_time:.1f} min\n"
                summary_text += f"Total Time:        {elapsed:.1f} min\n"
            
            ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')
        
        plt.tight_layout()
        save_path = self.output_dir / "training_monitor.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self):
        """Run monitoring loop"""
        print("="*60)
        print("TRAINING MONITOR STARTED")
        print("="*60)
        print(f"Monitoring directory: {self.checkpoint_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"\nPress Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                self.check_for_updates()
                self.plot_live_curves()
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            print(f"Final plots saved to: {self.output_dir}")


def main():
    # Configuration
    CHECKPOINT_DIR = "./checkpoints/stage1_foundation"
    OUTPUT_DIR = "./monitoring/stage1"
    UPDATE_INTERVAL = 60  # Update every 60 seconds
    
    monitor = TrainingMonitor(
        checkpoint_dir=CHECKPOINT_DIR,
        output_dir=OUTPUT_DIR,
        update_interval=UPDATE_INTERVAL
    )
    
    monitor.run()


if __name__ == "__main__":
    main()