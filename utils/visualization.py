"""Trajectory and loss visualization."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_trajectories(observed, generated, time_axis=None, save_path=None, title='Trajectories'):
    """Plot observed vs generated trajectories (single feature or multi)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if time_axis is None:
        time_axis = np.arange(observed.shape[0])
    if observed.ndim == 1:
        ax.plot(time_axis, observed, 'b-', label='Observed', alpha=0.8)
        ax.plot(time_axis, generated, 'r--', label='Generated', alpha=0.8)
    else:
        for f in range(observed.shape[1]):
            ax.plot(time_axis, observed[:, f], '-', label=f'Observed F{f}', alpha=0.8)
            ax.plot(time_axis, generated[:, f], '--', label=f'Generated F{f}', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss_curves(history, save_path=None):
    """Plot train/val loss from trainer.history."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history.get('train_loss', []), label='Train')
    ax.plot(history.get('val_loss', []), label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training loss')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
