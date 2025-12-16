"""
Generate All Project Images

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
import numpy as np

def generate_feature_importance_image():
    """Generate feature importance visualization image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='coolwarm', extent=[0, 10, 0, 8], alpha=0.2)
    
    # Title
    ax.text(5, 6.5, 'Feature Importance', fontsize=42, fontweight='bold',
            ha='center', va='center', color='#1976D2', family='sans-serif')
    ax.text(5, 5.8, 'XGBoost Analysis', fontsize=28, fontweight='bold',
            ha='center', va='center', color='#1565C0', family='sans-serif')
    
    # Features visualization (mock bars)
    features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
    importance = [0.25, 0.20, 0.18, 0.15, 0.12]
    
    y_start = 4.5
    for i, (feat, imp) in enumerate(zip(features, importance)):
        bar_width = imp * 8
        rect = mpatches.Rectangle((1, y_start - i*0.6), bar_width, 0.4,
                                 facecolor='steelblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.5, y_start - i*0.6 + 0.2, feat, ha='right', va='center',
               fontsize=12, fontweight='bold')
        ax.text(bar_width + 1.2, y_start - i*0.6 + 0.2, f'{imp:.2f}',
               ha='left', va='center', fontsize=11)
    
    # Footer
    ax.text(5, 0.8, 'RSK World - https://rskworld.in', fontsize=11,
            ha='center', va='center', color='#616161', family='sans-serif')
    
    plt.tight_layout()
    plt.savefig('feature_importance_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Generated: feature_importance_demo.png")
    plt.close()


def generate_hyperparameter_tuning_image():
    """Generate hyperparameter tuning visualization image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='viridis', extent=[0, 10, 0, 8], alpha=0.2)
    
    # Title
    ax.text(5, 7, 'Hyperparameter Tuning', fontsize=40, fontweight='bold',
            ha='center', va='center', color='#2E7D32', family='sans-serif')
    
    # Parameters
    params = [
        'max_depth: [3, 6, 9, 12]',
        'learning_rate: [0.01, 0.1, 0.2]',
        'n_estimators: [50, 100, 200]',
        'subsample: [0.8, 1.0]',
        'colsample_bytree: [0.8, 1.0]'
    ]
    
    y_start = 5.5
    for i, param in enumerate(params):
        ax.text(5, y_start - i*0.5, f'• {param}', fontsize=14,
               ha='center', va='center', color='#1B5E20', family='monospace')
    
    # Methods
    ax.text(5, 2.5, 'Methods: GridSearch | RandomizedSearch | Bayesian', 
           fontsize=16, ha='center', va='center', color='#424242',
           style='italic', fontweight='bold')
    
    # Footer
    ax.text(5, 0.8, 'RSK World - https://rskworld.in', fontsize=11,
            ha='center', va='center', color='#616161', family='sans-serif')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Generated: hyperparameter_tuning_demo.png")
    plt.close()


def generate_model_comparison_image():
    """Generate model comparison image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='plasma', extent=[0, 10, 0, 8], alpha=0.2)
    
    # Title
    ax.text(5, 7, 'Model Comparison', fontsize=40, fontweight='bold',
            ha='center', va='center', color='#7B1FA2', family='sans-serif')
    
    # Models
    models = [
        ('Shallow Model', 0.85),
        ('Medium Model', 0.92),
        ('Deep Model', 0.90),
        ('Ensemble', 0.94)
    ]
    
    y_start = 5.5
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    for i, (name, score) in enumerate(models):
        bar_width = score * 6
        rect = mpatches.Rectangle((2, y_start - i*0.8), bar_width, 0.5,
                                 facecolor=colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.8, y_start - i*0.8 + 0.25, name, ha='right', va='center',
               fontsize=13, fontweight='bold')
        ax.text(bar_width + 2.2, y_start - i*0.8 + 0.25, f'{score:.2%}',
               ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Footer
    ax.text(5, 0.8, 'RSK World - https://rskworld.in', fontsize=11,
            ha='center', va='center', color='#616161', family='sans-serif')
    
    plt.tight_layout()
    plt.savefig('model_comparison_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Generated: model_comparison_demo.png")
    plt.close()


def generate_learning_curve_image():
    """Generate learning curve visualization image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap='cool', extent=[0, 10, 0, 8], alpha=0.2)
    
    # Title
    ax.text(5, 7, 'Learning Curves', fontsize=40, fontweight='bold',
            ha='center', va='center', color='#00BCD4', family='sans-serif')
    
    # Mock learning curve
    epochs = np.linspace(1, 9, 20)
    train_loss = 0.5 * np.exp(-epochs/3) + 0.1
    val_loss = 0.6 * np.exp(-epochs/3) + 0.15
    
    # Scale to plot coordinates
    epochs_scaled = 1 + (epochs - 1) * 8 / 8
    train_scaled = 5 - (train_loss - 0.1) * 3 / 0.5
    val_scaled = 5 - (val_loss - 0.1) * 3 / 0.5
    
    ax.plot(epochs_scaled, train_scaled, 'o-', linewidth=3, markersize=6,
           label='Train Loss', color='#2196F3')
    ax.plot(epochs_scaled, val_scaled, 's-', linewidth=3, markersize=6,
           label='Validation Loss', color='#F44336')
    
    ax.text(5, 1.5, 'Epochs →', fontsize=12, ha='center', va='center', color='#424242')
    ax.text(0.5, 5, 'Loss ↓', fontsize=12, ha='center', va='center', 
           rotation=90, color='#424242')
    
    # Footer
    ax.text(5, 0.8, 'RSK World - https://rskworld.in', fontsize=11,
            ha='center', va='center', color='#616161', family='sans-serif')
    
    plt.tight_layout()
    plt.savefig('learning_curve_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Generated: learning_curve_demo.png")
    plt.close()


def main():
    """Generate all images."""
    print("=" * 60)
    print("Generating All Project Images")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    generate_feature_importance_image()
    generate_hyperparameter_tuning_image()
    generate_model_comparison_image()
    generate_learning_curve_image()
    
    print("\n" + "=" * 60)
    print("All images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

