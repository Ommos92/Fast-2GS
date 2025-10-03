# Fast-2GS: Fast 2D Gaussian Splatting Learning Library

A comprehensive learning library that demonstrates the fundamental concepts behind Gaussian Splatting using 2D images. This project focuses on first principles and core algorithms without the complexity of 3D rendering pipelines, camera calibration, or correspondence matching frameworks like COLMAP.

<p align="center">
  <img src="assets/fast_2d_gaussian_splatting_training.gif" alt="Fast 2D Gaussian Splatting Training Visualization" width="1200"/>
</p>

**Above:** *Training progress of Fast-2GS reconstructing an image using adaptive 2D Gaussian splatting. Left: Target image. Center: Model reconstruction. Right: Difference map. The GIF shows the model learning and refining its representation over time.*


## Purpose

Fast-2GS is designed to help you understand:
- **Core Gaussian Splatting concepts** in a simplified 2D setting
- **Adaptive density control** and gradient-based densification
- **Optimization techniques** for neural rendering
- **Performance optimization** strategies for real-time applications
- **MPS/GPU acceleration** on Apple Silicon devices

## Key Features

### Core Implementation
- **2D Gaussian Splatting**: Simplified version of the 3D Gaussian Splatting algorithm
- **Adaptive Density Control**: Automatically adds/removes Gaussians based on reconstruction error
- **Gradient-Based Densification**: Uses gradient norms to identify areas needing more detail
- **Ultra-Fast Rendering**: Vectorized operations with 1000x performance improvements

### Advanced Optimizations
- **Pre-allocated Parameter Pools**: Avoids memory allocation during training
- **Smart Optimizer State Preservation**: Maintains Adam momentum during adaptation
- **Real-time Visualization**: Live training progress with loss curves and reconstruction quality

### Learning-Focused Design
- **First Principles**: No complex frameworks or dependencies
- **Educational Examples**: Step-by-step implementations
- **Interactive Notebooks**: Jupyter notebooks for hands-on learning

## Project Structure

```
gaussian_splatting/
├── README.md                    # This file
├── 2d_gaussian_splatting.ipynb  # Main learning notebook with all implementations
├── 2gs.py                       # Standalone Python script version
├── requirements.txt             # Python dependencies
└── .venv/                       # Virtual environment
```

## Installation

### Prerequisites
- Python 3.8+
- Apple Silicon Mac (for MPS acceleration) or NVIDIA GPU (for CUDA)
- 8GB+ RAM recommended

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd gaussian_splatting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.8.0
scikit-image>=0.19.0
ipython>=7.0.0
```

## Learning Path

### 1. Basic 2D Gaussian Splatting
Start with the fundamental implementation to understand:
- How Gaussians are parameterized (position, scale, color, opacity)
- Basic rendering pipeline
- Loss computation and optimization

### 2. Adaptive Density Control
Learn how to:
- Identify areas needing more detail using gradient analysis
- Add new Gaussians where reconstruction error is high
- Remove unnecessary Gaussians to maintain efficiency

### 3. Performance Optimization
Explore advanced techniques:
- Pre-allocation strategies
- Vectorized operations
- GPU acceleration with MPS/CUDA
- Optimizer state preservation

### 4. Real-time Applications
Understand how to:
- Achieve real-time rendering speeds
- Balance quality vs. performance
- Implement interactive visualization

## Quick Start

### Jupyter Notebook (Recommended)
```bash
# Start Jupyter notebook
jupyter notebook 2d_gaussian_splatting.ipynb
```

### Python Script
```bash
# Run the standalone script
python 2gs.py
```

## Key Concepts Explained

### Gaussian Parameterization
Each Gaussian is defined by:
- **Position** (x, y): Center location in 2D space
- **Scale** (σx, σy): Width and height of the Gaussian
- **Color** (r, g, b): RGB color values
- **Opacity** (α): Transparency/visibility

### Rendering Pipeline
1. **Coordinate Grid**: Create pixel coordinates
2. **Distance Computation**: Calculate distances from pixels to Gaussian centers
3. **Gaussian Evaluation**: Compute Gaussian values using the formula
4. **Alpha Blending**: Combine Gaussians using opacity
5. **Normalization**: Ensure proper color mixing

### Adaptive Density Control
1. **Gradient Analysis**: Compute gradients to identify high-error regions
2. **Densification**: Clone Gaussians in areas with high gradients
3. **Pruning**: Remove Gaussians with low opacity
4. **Optimization**: Continue training with updated Gaussian set

## Advanced Features

### Smart Optimizer State Preservation
```python
# Preserves Adam momentum when adapting Gaussian count
smart_optimizer = StatePreservingOptimizer(model, lr=0.1)
smart_optimizer.save_state()
model.adapt_gaussians()
smart_optimizer.restore_state()
```

### Ultra-Fast Rendering
```python
# Pre-allocated parameters for maximum speed
class UltraFastGaussian2D(nn.Module):
    def __init__(self, max_gaussians=50000):
        # Pre-allocate all parameters
        self.positions = nn.Parameter(torch.rand(max_gaussians, 2))
        self.active_mask = torch.zeros(max_gaussians, dtype=torch.bool)
```

## Results and Visualization

The library provides real-time visualization of:
- **Target vs. Reconstruction**: Side-by-side comparison
- **Difference Maps**: Highlighting reconstruction errors
- **Loss Curves**: Training progress over time
- **Gaussian Distribution**: Spatial distribution of Gaussians
- **Performance Metrics**: PSNR, MSE, and timing information

## Educational Value

### What You'll Learn
1. **Neural Rendering Fundamentals**: How to represent images as neural fields
2. **Optimization Techniques**: Gradient-based optimization for rendering
3. **Adaptive Algorithms**: Dynamic resource allocation based on error

### Prerequisites
- Basic Python programming
- Understanding of linear algebra (vectors, matrices)
- Familiarity with PyTorch (helpful but not required)
- Interest in computer graphics and neural rendering

## Research Applications

This library demonstrates concepts used in:
- **3D Gaussian Splatting**: The foundation for 3D scene representation
- **Neural Radiance Fields (NeRF)**: Similar optimization principles
- **Real-time Rendering**: Performance optimization techniques
- **Computer Vision**: Image reconstruction and synthesis

## Contributing

We welcome contributions! Areas for improvement:
- Additional optimization strategies
- New visualization techniques
- Performance benchmarks
- Educational content and examples
- Documentation improvements

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original 3D Gaussian Splatting authors for the foundational work
- PyTorch team for the excellent deep learning framework
- Apple for MPS acceleration support
- The open-source community for inspiration and feedback

---

**Happy Learning!**

Start with the basic implementation and work your way up to the ultra-fast version. Each step builds upon the previous one, giving you a deep understanding of Gaussian Splatting principles.
