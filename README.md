# 🎯 OtakuLens AI: Hybrid Deep Feature Extraction Pipeline
LINK -> https://otakulens-ai.streamlit.app/

Welcome to **OtakuLens AI**! I built this end-to-end machine learning project to tackle a notoriously difficult computer vision task: **classifying 2D 
anime and manga characters**. 

Unlike real-world photographs, 2D line art lacks realistic shadows, depth, and textures, which often causes standard Convolutional Neural Networks (CNNs) to fail. This repository documents my journey of taking a broken, overfitting model (21% accuracy) and engineering a robust **PyTorch + XGBoost Hybrid Pipeline** to achieve **85%+ production accuracy** across 15 distinct character classes.

---

## 🧠 The Engineering Challenge

When I first built this classifier, I ran into massive data friction. I hit what I called the **"Luffy Anomaly"**—my model collapsed into a local minimum and just started guessing "Luffy" for every image to artificially lower its loss function. 

I traced the failure back to three core ML architectural flaws:
1. **Severe Data Starvation:** I only had a few dozen images per character.
2. **Data Leakage:** I was applying PCA dimensionality reduction to the *entire* dataset before making my Train/Test split, allowing test variance to leak into the training loop.
3. **Backbone Blindness:** Standard ResNet models are trained on ImageNet (real-world items like cars and dogs), rendering them somewhat blind to 2D sketch vectors.

## 🛠️ The Architecture (How I Fixed It)

To solve these issues, I completely rebuilt the pipeline into a hybrid deep-feature extraction system:

* **Dynamic Data Augmentation (Albumentations):** To solve the data starvation without bloating my hard drive, I implemented a 20-epoch multi-pass extraction loop. Using heavy random transformations (pixel dropout, color jitter, harsh rotation), I forced a frozen feature extractor to process completely unique mathematical signatures, scaling my dataset volume **20x**.
* **Deep Feature Extraction (PyTorch ResNet-50):** I passed the augmented images through a pre-trained ResNet-50 backbone. I unfroze the deepest convolutional blocks (`layer4`) to allow the filters to shift explicitly to black/white art vectors, outputting 2048 raw deep features per image.
* **Leak-Proof Dimensionality Reduction (Scikit-Learn):** I enforced a strict data split *first*. I then fit a `StandardScaler` and `PCA` (retaining 95% variance) strictly on the training indices. This compressed the 2048 noisy features down to ~120 highly potent structural components.
* **The Ultimate Classifier (XGBoost):** Traditional SVMs struggled with the high-dimensional tabularized vectors. I swapped the classification head to an XGBoost tree-ensemble, which masterfully handled the non-linear relationships in the compressed feature space.

## 📊 Results & Confusion Matrix

**Initial Baseline Accuracy:** 21.0%
**Final Production Accuracy:** ~85.4%

The model now successfully differentiates between visually similar tropes (e.g., distinguishing Akira Toriyama's identical eye-shapes for Goku and Vegeta). The finalized model weights and PCA parameters are serialized via `joblib` for rapid deployment.

---

## 💻 Streamlit Web Application

I deployed this pipeline into a fully interactive Streamlit web application. 

**Features:**
* **🎮 Active Inference Engine:** A "conveyor belt" UI allowing users to click sample images and run live, on-the-fly math transformations and XGBoost predictions.
* **📚 Dataset Vault:** A structured visual catalog to audit the raw image arrays.
* **🧠 Architect Logs:** An "interview cheat-sheet" tab where I break down the intuition, dataset transformations, and performance metrics dynamically using Matplotlib and Seaborn.
