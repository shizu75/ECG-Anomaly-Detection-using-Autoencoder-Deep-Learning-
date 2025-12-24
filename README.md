# ECG Anomaly Detection using Autoencoder (Deep Learning)

## Overview

This repository implements an **unsupervised deep learning–based anomaly detection system** for **electrocardiogram (ECG) signals** using an **Autoencoder neural network**. The model is trained exclusively on **normal heartbeats** and learns to reconstruct them accurately. Abnormal ECG signals are detected based on **reconstruction error**, making this approach suitable for real-world anomaly detection where labeled anomalies are scarce.

This project demonstrates a **biomedical signal processing + deep learning workflow**, aligning well with research in **biomedical engineering, AI for healthcare, and physiological signal analysis**.

---

## Dataset

- **Source**: TensorFlow public dataset  
- **Samples**: ECG signals with 140 time points per sample
- **Labels**:
- `1` → Normal heartbeat
- `0` → Abnormal heartbeat

Each row consists of:
- 140 ECG signal values
- 1 binary label (last column)

---

## Data Preprocessing

### 1. Train–Test Split
- 80% training data
- 20% testing data

### 2. Normalization
Min–max normalization applied using training data statistics:
\[
x_{norm} = \frac{x - \min}{\max - \min}
\]

### 3. Data Separation
- **Normal signals** → used for training the autoencoder
- **Anomalous signals** → used only for testing

This enforces a **true unsupervised learning setup**.

---

## Exploratory Data Analysis

- Visualization of:
- Normal vs abnormal ECG signals
- Mean ECG waveform for each class
- Highlights structural differences between healthy and anomalous heartbeats

---

## Model Architecture

### Autoencoder Design

The model consists of two symmetric components:

#### Encoder
- Dense (32 units, ReLU)
- Dense (16 units, ReLU)
- Dense (8 units, ReLU)

#### Decoder
- Dense (16 units, ReLU)
- Dense (32 units, ReLU)
- Dense (140 units, Sigmoid)

The encoder compresses the ECG signal into a low-dimensional latent representation, while the decoder reconstructs the signal.

---

## Training Configuration

- **Loss Function**: Mean Absolute Error (MAE)
- **Optimizer**: Adam
- **Epochs**: 20
- **Batch Size**: 512
- **Training Data**: Normal ECG signals only
- **Validation Data**: Full test set

Training and validation loss curves are plotted to monitor convergence.

---

## Reconstruction Analysis

- Visual comparison of:
- Input ECG signal
- Reconstructed ECG
- Reconstruction error area
- Separate plots for:
- Normal ECG reconstruction
- Abnormal ECG reconstruction

Normal signals show low reconstruction error, while abnormal signals exhibit visibly higher error.

---

## Anomaly Detection Logic

### Reconstruction Error
\[
\text{Error} = \text{MAE}(\text{input}, \text{reconstruction})
\]

### Threshold Selection
\[
\text{Threshold} = \mu_{train\_loss} + \sigma_{train\_loss}
\]

Any ECG signal with reconstruction loss **above the threshold** is classified as **anomalous**.

---

## Evaluation Metrics

The model is evaluated on the test dataset using:

- **Accuracy**
- **Precision**
- **Recall**

Predictions are generated using threshold-based reconstruction loss comparison.

---

## Results & Observations

- Clear separation between reconstruction loss distributions of normal and anomalous signals
- High recall indicates strong anomaly detection capability
- Demonstrates effectiveness of autoencoders for biomedical anomaly detection

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Research Significance

This project showcases:
- Unsupervised learning for biomedical signals
- Practical anomaly detection without reliance on labeled abnormal data
- A scalable approach applicable to ECG, EEG, EMG, and other biosignals

It is suitable for:
- Biomedical engineering portfolios
- AI-for-healthcare research
- Graduate and PhD-level projects

---

## Disclaimer

This project is for **educational and research purposes only** and is not intended for clinical diagnosis or medical decision-making.

---

## Author Note

This work reflects a **research-oriented implementation of deep learning for physiological signal analysis**, emphasizing interpretability, methodology, and biomedical relevance.
