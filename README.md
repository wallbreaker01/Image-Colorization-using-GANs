# Image Colorization using GANs

A deep learning project that uses Generative Adversarial Networks (GANs) to automatically colorize grayscale landscape images.

## Overview

This project implements a Pix2Pix-style GAN architecture to transform grayscale landscape images into realistic colored images. The model learns the mapping between grayscale and color images through adversarial training.

## Features

- **Generator**: U-Net architecture with encoder-decoder structure and skip connections
- **Discriminator**: PatchGAN discriminator for realistic image classification
- **Dataset**: Landscape images dataset with paired grayscale and color images
- **Training**: Adversarial training with combined GAN loss and L1 loss

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV (cv2)
- scikit-learn
- tqdm
- kagglehub

## Installation

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn tqdm kagglehub
```

## Dataset

The project uses the [Landscape Image Colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) dataset from Kaggle, which contains:
- Color landscape images
- Corresponding grayscale versions
- 256x256 resolution images

## Model Architecture

### Generator (U-Net)
- 8 downsampling layers (encoder)
- 7 upsampling layers (decoder)
- Skip connections between encoder and decoder
- Output: 256x256x3 colored image

### Discriminator (PatchGAN)
- Classifies 70x70 patches as real or fake
- Takes both input (grayscale) and target/generated (color) images
- Helps ensure realistic colorization

## Training

The model is trained using:
- **Batch size**: 64 for training, 8 for testing
- **Epochs**: 50
- **Optimizer**: Adam (learning rate: 2e-4, beta_1: 0.5)
- **Loss function**: 
  - Generator: Binary cross-entropy + L1 loss (λ=100)
  - Discriminator: Binary cross-entropy

## Usage

1. Open the Jupyter notebook `Image Colorization.ipynb`
2. Run the cells sequentially to:
   - Load and preprocess the dataset
   - Build the Generator and Discriminator models
   - Train the GAN
   - Generate colorized images from grayscale inputs
   - Visualize results and training progress

## Results

The model generates colorized versions of grayscale landscape images. Training progress can be monitored through:
- Generator and Discriminator loss curves
- Sample image generation during training
- Visual comparison of input, ground truth, and predicted images

## Project Structure

```
Image-Colorization-using-GANs/
├── Image Colorization.ipynb  # Main notebook with complete implementation
└── README.md                  # Project documentation
```

## Acknowledgments

- Dataset from Kaggle: [Landscape Image Colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)
- Based on Pix2Pix GAN architecture