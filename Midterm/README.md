# 🖼️ Midterm Project: Image Classification using ResNet50 (Transfer Learning)

## 📌 Overview

This project demonstrates the use of a **pretrained convolutional neural network (CNN)** — ResNet50 — for **image classification**. It showcases the application of **AI in computer vision** using **PyTorch** and transfer learning.

> 🔍 **Category**: AI Application (Image Recognition)  
> 📖 **Type**: Original implementation using reference-based adaptation (no copy/paste)  
> ✨ **AI Support**: Explanation, documentation, and structure guided by ChatGPT

---

## 🎯 Objectives

- Use a pretrained model (`ResNet50`) from `torchvision.models`
- Apply **transfer learning** to classify custom input images
- Understand the structure and inference process of CNNs

---

## 🏗️ How It Works

1. Load the pretrained ResNet50 model
2. Preprocess an input image (resize, normalize, convert to tensor)
3. Feed it into the model and get predictions
4. Match output logits to **ImageNet class labels**

---

## 📦 Requirements

```bash
pip install torch torchvision pillow


python image_classification_resnet50.py
