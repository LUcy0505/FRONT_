# FRONT
## [Title] Breaking the Scale Barrier: One-Shot Knowledge Transfer via Frequency Transform

🎉 **Accepted to ICML 2026!** 
> This repository contains the official PyTorch implementation of the paper *"Breaking the Scale Barrier: One-Shot Knowledge Transfer via Frequency Transform"*, accepted at the International Conference on Machine Learning (ICML) 2026.

FRONT is a novel, cross-architecture initialization framework that leverages Frequency Domain Knowledge Transfer. By utilizing Discrete Cosine Transform (DCT) and Inverse Discrete Cosine Transform (IDCT), FRONT enables seamless transfer of structural knowledge from a large, pre-trained base model (source) to initialize a smaller, target model. 

This initialization method smooths out the feature alignment process and accelerates convergence for models like Vision Transformers (e.g., DeiT), fundamentally serving as a superior starting point compared to random initialization.

## ✨ Core Concept

FRONT does not train models from scratch. Instead, it **initializes the target model** by extracting low-frequency, high-energy components from a well-trained source model. 

1. **Extraction:** It applies DCT to the weights of the source model (`basemodel`) to obtain frequency domain coefficients.
2. **Alignment & Truncation:** It filters the parameters (controlling the preservation ratio) and maps the depth of the networks.
3. **Reconstruction:** It applies IDCT to reconstruct the weights and injects them directly into the target model (`model`) in memory.
4. **Output:** A customized, dynamically initialized model ready for fine-tuning.

## 🚀 Usage Guide

The provided script performs the FRONT initialization pipeline in memory and saves the initialized weights for the target model. 

### Basic Execution

```bash
python front_init.py \
  --dct \
  --use_dct \
  --basemodel deit_tiny_patch16_224_L12 \
  --model deit_tiny_patch16_224_L4 \
  --basemodel_pretrain_pth path/to/base_weights.pth \
  --keep_ratio 1.0 \
  --mlp_dct_dim 2d \
  --dct_mode zero \
  --output_dir checkpoints/
