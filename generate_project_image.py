"""
Generate Project Image Placeholder

Project: XGBoost Gradient Boosting
Author: Molla Samser (Founder)
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
GitHub: https://github.com/rskworld
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Background gradient
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, aspect='auto', cmap='viridis', extent=[0, 10, 0, 8], alpha=0.3)

# Title
ax.text(5, 6.5, 'XGBoost', fontsize=48, fontweight='bold', 
        ha='center', va='center', color='#2E7D32', family='sans-serif')
ax.text(5, 5.5, 'Gradient Boosting', fontsize=32, fontweight='bold',
        ha='center', va='center', color='#1B5E20', family='sans-serif')

# Subtitle
ax.text(5, 4.5, 'Advanced Machine Learning with XGBoost', fontsize=18,
        ha='center', va='center', color='#424242', style='italic')

# Features
features = [
    '✓ Hyperparameter Optimization',
    '✓ Feature Importance Analysis',
    '✓ Cross-Validation Techniques',
    '✓ Model Interpretation'
]

y_start = 3.5
for i, feature in enumerate(features):
    ax.text(5, y_start - i * 0.4, feature, fontsize=14,
            ha='center', va='center', color='#212121', family='sans-serif')

# Footer
ax.text(5, 0.8, 'RSK World - https://rskworld.in', fontsize=12,
        ha='center', va='center', color='#616161', family='sans-serif')
ax.text(5, 0.4, 'Molla Samser (Founder) | Rima Khatun (Designer & Tester)', 
        fontsize=10, ha='center', va='center', color='#757575', family='sans-serif')

# Decorative elements
# Add some geometric shapes
circle1 = plt.Circle((1.5, 6.5), 0.3, color='#4CAF50', alpha=0.6)
circle2 = plt.Circle((8.5, 6.5), 0.3, color='#66BB6A', alpha=0.6)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Save
plt.tight_layout()
plt.savefig('xgboost-boosting.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Project image saved as 'xgboost-boosting.png'")
plt.close()

