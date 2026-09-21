# Comparative Study of DCGAN and WGAN-GP on CelebA

Comparison of **DCGAN** and **WGAN-GP** for face generation on the **CelebA dataset**, with a focus on training stability, output quality, and hyperparameter sensitivity.

## Results Preview

<p align="center">
  <img width="975" alt="DCGAN vs WGAN-GP comparison"
       src="https://github.com/user-attachments/assets/c9c68d4a-2d5b-4b54-839f-49a175e46a50" />
</p>

<p align="center">
  <img width="1059" alt="Generated samples and training results"
       src="https://github.com/user-attachments/assets/e5193e43-9a39-4ba0-9b53-2aabc1ec83b0" />
</p>

---

## Overview

This project compares two generative adversarial network architectures:

- **DCGAN**
- **WGAN-GP**

Both models were trained on a subset of **10,000 CelebA images at 64×64 resolution**.

The experiments focus on:

- training stability
- generated image quality
- mode collapse
- sensitivity to batch size and learning rate
- effect of critic updates and gradient penalty in WGAN-GP

---

## Key Findings

### DCGAN

- Learns facial structure quickly
- Can generate recognizable faces after relatively few epochs
- More sensitive to learning rate and batch size
- Occasionally suffers from unstable training and mode collapse

### WGAN-GP

- Shows smoother and more stable training behavior
- Produces more diverse samples
- Gradient penalty improves stability compared with standard GAN training
- Performance depends strongly on critic updates and λ

---

## Features

- PyTorch implementations of DCGAN and WGAN-GP
- CelebA subset sampling
- 64×64 image generation
- Generated sample visualization
- Loss curve visualization
- Configurable hyperparameters
- Checkpointing and resume support

---

## Project Structure

```text
.
├── train.py
├── train_*.py
├── samples/
├── checkpoints/
└── requirements.txt
