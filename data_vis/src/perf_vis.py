import matplotlib.pyplot as plt
import json

# Load training history and test metrics
with open('model_checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

# Configure metrics to plot (no need for test metrics)
metrics_config = [
    ('dice', 'val_dice', 'Dice Score'),
    ('segmentation_iou_metric', 'val_segmentation_iou_metric', 'Segmentation IoU'),
    ('mae', 'val_mae', 'MAE'),
    ('seg_loss', 'val_seg_loss', 'Segmentation Loss'),
    ('lm_loss', 'val_lm_loss', 'Landmark Loss'),
    ('total_loss', 'val_total_loss', 'Total Loss')
]

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(18, 15))
axs = axs.flatten()

# Plot each metric
for idx, (train_key, val_key, title) in enumerate(metrics_config):
    ax = axs[idx]
    epochs = range(1, len(history[train_key]) + 1)

    # Plot training and validation curves
    ax.plot(epochs, history[train_key], 'b-', label='Training')
    ax.plot(epochs, history[val_key], 'g-', label='Validation')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epochs', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout(pad=3.0)
plt.savefig('data_vis/results/training_metrics_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
