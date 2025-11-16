---

# Comparative Study of DCGAN and WGAN-GP on CelebA

This repository contains training scripts and experiments comparing **DCGAN** and **WGAN-GP** for image generation using the **CelebA dataset**.
The goal is to evaluate training stability, output quality, and the effect of different hyperparameters on generative performance.

---

## Features

* Training pipelines for DCGAN and WGAN-GP
* Subset sampling support (10k CelebA images, 64×64 resolution)
* Visualization of generated samples and loss curves
* Modular and configurable training scripts (PyTorch)
* Checkpointing and resume training support

---

## Folder Structure

* `train_*.py` – Training scripts for different GAN configurations
* `samples/` – Generated sample images (to be uploaded)
* `checkpoints/` – Saved model weights and optimizer states

---

## Usage

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/dcgan-wgan-celeba.git
cd dcgan-wgan-celeba
pip install -r requirements.txt
```

Run training:

```bash
# Train DCGAN
python train.py --model dcgan --epochs 10 --batch_size 128

# Train WGAN-GP
python train.py --model wgan_gp --epochs 5 --batch_size 64 --n_critic 5
```

Generated images and logs will be saved under `samples/`.

---

## Results

* **DCGAN**

  * Learns facial structure over epochs, but sometimes suffers from mode collapse
  * Sensitive to batch size and learning rate

* **WGAN-GP**

  * More stable training dynamics with Wasserstein loss and gradient penalty
  * Produces more diverse and realistic faces

Example results (DCGAN vs WGAN-GP):
<img width="1059" height="416" alt="image" src="https://github.com/user-attachments/assets/e5193e43-9a39-4ba0-9b53-2aabc1ec83b0" />


---

## Conclusion

* DCGAN is simple and effective but prone to instability
* WGAN-GP converges more smoothly and generates higher-quality outputs
* Hyperparameter tuning (batch size, critic updates, λ) is critical for stability

Future work:

* Scaling to higher resolutions (128×128 or beyond)
* Exploring advanced architectures such as StyleGAN or Diffusion Models

---

## References

* [PyTorch DCGAN Example](https://github.com/pytorch/examples/tree/main/dcgan)
* [WGAN-GP Implementation](https://github.com/caogang/wgan-gp)
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---


