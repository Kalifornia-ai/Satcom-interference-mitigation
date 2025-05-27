# CW Power Estimation from Raw RF Bursts

This repository contains the code, models, and utilities for estimating the power of a narrowband continuous-wave (CW) signal buried in broadband interference, such as QPSK, using deep learning.

The trained models predict the complex gain of the CW tone from raw, 1000-sample I/Q bursts without requiring frequency-domain processing. This approach is designed for beacon tracking, radiation pattern verification, and SDR-based link calibration in co-channel interference scenarios.

---

## 🔍 Overview
- 'generateSimulatedData.py'
  Generates simulated data following a normal signal propagation path from generation to receiving and I/Q measurement

- `createGain.py`  
  Builds a structured dataset of `.npz` bursts by combining mixture and clean-tone recordings. Computes complex gain from aligned CW bursts.

- `makePlots.py`  
  Visualizes mixture, clean, and estimated signals in time, frequency (PSD), and I/Q (constellation) domains for each (CW, QPSK) power pair.

- `model.py`  
  Trains a compact CNN model (`BeaconPowerCNN`) to estimate CW gain in dBm. Includes training, validation, test evaluation, and loss curves.

  - `train_hybrid.py`  
  Trains a compact Residual model (`HybridBeaconEstimator`) to estimate CW gain in dBm. Includes training, validation, test evaluation, and loss curves.

  - `train_lstm.py`  
  Trains a compact  model (`LSTMSeperatorSingle`), ('LSTMSingleSource'), to estimate CW gain in dBm. Includes training, validation, test evaluation, and loss curves.
  
  - 'train_causalLSTM.py'
  Trains a compact CausalLSTM model of output dimension (B,1000,2) to estimate the real sinusoid waveform. Includes training, validation, test evaluation.
- `resmodel.py` / `lstm_model.py`  
  Contains alternative models including HybridBeaconEstimator, LSTMSeperatorSingle, and LSTMSingleSource.

---

## 🗂 Dataset Structure

Each processed `.npz` file contains:

```python
x     # np.complex64, shape (1000,) – raw frequency-shifted burst
meta  = {
  'pristine_gain': complex,     # ground-truth CW tone gain
  'offset_hz': 200_000,         # coarse shift applied
  'fs': 10_000_000.0            # sampling rate
}

## Model Architecture

BeaconPowerCNN
Input: 2 × 1000 (real, imag)

3 × Conv1D layers + ReLU

AdaptiveAvgPool + 2-layer MLP

Output: 2-dim vector representing Re(g), Im(g)

Other models include:

HybridBeaconEstimator (ResNet + LSTM) - given name is DC-CRL

LSTMSeperatorSingle (stacked BiLSTM + SE blocks) - given name is Bi-LSTM

LSTMSingleSource (SE + masking + output scaling - given name is CausalLSTM_SingleOutput


## Training

python model.py ./dataset \
  --epochs 200 --batch 16 --lr 2e-4 \
  --best-name best_beacon_cnn.pt

  python train_hybrid.py ./dataset \
  --epochs 200 --batch 16 --lr 2e-4 \
  --best-name hybrid_cnn.pt

    python train_lstm.py ./dataset \
  --epochs 200 --batch 16 --lr 2e-4 \
  --best-name lstm_cnn.pt

## Evaluation and Visualization

python eval.py ./dataset  --ckpt ./dataset/best_beacon_cnn.pt

  This will produce:


Each shows absolute error, box plots, and model-estimated CW.



All code is released under the MIT License.
© Kadyrzhan Tortayev, 2025.
