# 🚀 Projective FHE GPU Benchmark on Google Colab

Complete setup guide to benchmark the projective FHE system with CUDA acceleration on Google Colab.

## 🎯 Expected Results

- **10-20× GPU speedup** over CPU implementation
- **4096-bit mode**: ~10-12ms per mult→project cycle (GPU) vs ~114ms (CPU)
- **8192-bit mode**: ~20-24ms per mult→project cycle (GPU) vs ~228ms (CPU)
- **Interactive FHE**: Sub-15ms cycles enable real-time applications
- **High throughput**: 40-80 cycles/second depending on ring dimension

## 📋 Prerequisites

1. **Google Account** with Google Drive access
2. **Google Colab Pro** (recommended for better GPU access and longer runtimes)
3. **GPU runtime**: Must switch to GPU before running

## 🚀 Quick Start (One-Click Setup)

### Option 1: Single Bash Script (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. **Switch to GPU runtime**: Runtime → Change runtime type → GPU → Save
3. Create a new code cell and paste:

```bash
# Download and run the complete setup script
!curl -fsSL https://raw.githubusercontent.com/franzwollang/projective-FHE/main/FHE/code/colab_gpu_setup.sh | bash
```

That's it! The script will:

- ✅ Mount Google Drive and create workspace
- ✅ Install build dependencies
- ✅ Build OpenFHE with CUDA support
- ✅ Clone and build the projective FHE prototype
- ✅ Run GPU benchmark and save results to Drive

**Runtime**: ~10-15 minutes total

### Option 2: Step-by-Step Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. **Switch to GPU runtime**: Runtime → Change runtime type → GPU → Save
3. Upload the notebook: `FHE/code/colab_gpu_benchmark.ipynb`
4. Run all cells sequentially

## 📊 What Gets Generated

All files are automatically saved to `Google Drive/projective_fhe_benchmark/`:

- **`gpu_benchmark_results_YYYYMMDD_HHMMSS.txt`**: Complete benchmark output
- **`benchmark_summary_YYYYMMDD_HHMMSS.md`**: Performance summary and analysis
- **`gpu_benchmark_plot_YYYYMMDD_HHMMSS.png`**: Visualization comparing GPU vs CPU

## 🔧 Manual Setup (Advanced)

If you need to customize the build or troubleshoot:

### 1. Mount Drive & Setup Workspace

```python
from google.colab import drive
import os

drive.mount('/content/drive')
workspace = '/content/drive/MyDrive/projective_fhe_benchmark'
os.makedirs(workspace, exist_ok=True)
os.chdir(workspace)
```

### 2. Install Dependencies

```bash
!sudo apt-get update -qq
!sudo apt-get install -y build-essential cmake ninja-build git libomp-dev
!nvcc --version  # Verify CUDA
```

### 3. Build OpenFHE with CUDA

```bash
!git clone --depth 1 https://github.com/openfheorg/openfhe-development.git
%cd openfhe-development/build
!cmake .. -DWITH_GPU=ON -DCUDA_ARCHITECTURES=75 -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_UNITTESTS=OFF
!make -j4 && sudo make install
```

### 4. Build Projective FHE

```bash
!git clone https://github.com/franzwollang/projective-FHE.git
%cd projective-FHE/FHE/code/openfhe_prototype/build_gpu
!cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DIAGNOSTICS=OFF -DUSE_OPENFHE_GPU=ON
!make benchmark_modes -j4
```

### 5. Run Benchmark

```bash
!./benchmark_modes | tee results.txt
```

## 🐛 Troubleshooting

### Common Issues

**"No GPU available"**

- Ensure you switched runtime to GPU: Runtime → Change runtime type → GPU → Save
- Restart runtime if needed

**"OpenFHE build fails"**

- Reduce parallel jobs: `make -j2` instead of `make -j4`
- Check CUDA architecture: `nvidia-smi --query-gpu=compute_cap --format=csv`

**"benchmark_modes not found"**

- Check build succeeded: `ls -la benchmark_modes`
- Verify linking: `ldd benchmark_modes | grep openfhe`

**"Out of memory during build"**

- Use fewer cores: `make -j2` or `make -j1`
- Restart runtime to free memory

### Performance Verification

Expected output patterns:

```
🎯 4096-bit mode: 10.2 ms/cycle, 98.0 cycles/sec
🎯 8192-bit mode: 22.1 ms/cycle, 45.2 cycles/sec
✅ GPU acceleration: 11.2x speedup over CPU
```

If you see much slower times, GPU acceleration may not be working.

## 📈 Performance Analysis

### Benchmark Interpretation

- **Latency < 15ms**: Excellent for interactive applications
- **Throughput > 40 c/s**: Good for batch processing
- **Speedup > 10x**: GPU acceleration working properly
- **Noise levels ~15-18 RMS**: Projection working correctly

### Comparison with State-of-the-Art

| Metric          | Projective FHE (GPU)  | Traditional FHE           |
| --------------- | --------------------- | ------------------------- |
| Mult latency    | **10-24ms**           | ~1000ms (with bootstrap)  |
| Noise mgmt      | Projection (built-in) | Bootstrapping (expensive) |
| Circuit depth   | Unlimited\*           | Limited by noise budget   |
| Interactive use | ✅ Enabled            | ❌ Too slow               |

\*With occasional bootstrapping for very deep circuits

## 🎯 Next Steps

After successful benchmarking:

1. **Download results** from Google Drive
2. **Compare parameters** across different service tiers
3. **Scale testing** to larger k/p values
4. **Deploy** on dedicated GPU infrastructure
5. **Integrate** with your FHE applications

## 🚀 Production Deployment

For production use:

- **Hardware**: NVIDIA A100/H100 recommended
- **Memory**: 40GB+ GPU RAM for large parameters
- **Network**: High-bandwidth for client-server communication
- **Security**: Implement proper key management and attestation

---

**Repository**: https://github.com/franzwollang/projective-FHE  
**Paper**: See `FHE/reference/fhe_v3.md` for mathematical foundations  
**Issues**: Report problems via GitHub Issues
