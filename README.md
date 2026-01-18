# voice-command-classification-ai

This project implements a complete **Voice Detection and Recognition System** using MATLAB. It features a full pipeline from digital signal preprocessing to real-time inference using Machine Learning (KNN) and Deep Learning (Neural Networks) to detect and classify voice commands.

<br>

## Key Features

* **Signal Preprocessing**: Implementation of **Butterworth digital filters** for noise reduction and **MFCC (Mel-Frequency Cepstral Coefficients)** extraction to represent audio features efficiently.
* **Machine Learning Models**: Comparative study between **K-Nearest Neighbors (KNN)** and **Pattern Recognition Neural Networks** to identify the most robust architecture.
* **Performance Optimization**: Extensive evaluation across varied training datasets to minimize overfitting and maximize prediction accuracy.
* **Real-time Inference**: Designed a high-efficiency, low-latency loop capable of processing live microphone input for instantaneous command detection.
* **System Impact**: Successfully integrated the recognition engine as a control layer for interactive software applications.

<br>

## System Architecture

The project follows a standard AI-Audio pipeline:
1. **Acquisition**: Live audio capture via microphone or dataset loading.
2. **Feature Engineering**: Raw audio signals are filtered and transformed into MFCC feature vectors (reducing data dimensionality while preserving vocal characteristics).
3. **Classification Engine**: The feature vectors are fed into the trained models (KNN or NN) to output a command label.
4. **Control Loop**: The detected command triggers specific actions within the integrated application.

<br>

## Software Stack

* **MATLAB**: Main development environment.
* **Signal Processing Toolbox**: Used for digital filtering and spectral analysis.
* **Deep Learning & Statistics Toolbox**: For training and evaluating the Neural Network and KNN models.
* **Audio Toolbox**: For low-latency real-time microphone stream management.

<br>

## How to Run

1.  **Environment**: Open MATLAB (ensure Signal Processing and Deep Learning toolboxes are installed).
2.  **Training**: Run `train_models.m` to preprocess the dataset and train the KNN/Neural Network classifiers.
3.  **Real-time Detection**: Execute `live_inference.m` to start the microphone acquisition loop and see live predictions in the command window.
4.  **Evaluation**: Check `performance_metrics.m` to view confusion matrices and accuracy reports.

<br>

## Performance Results

* **Feature Extraction**: MFCCs proved significantly more effective than raw wave analysis for distinguishing between similar-sounding commands.
* **Model Comparison**: While KNN offered simplicity, the **Pattern Recognition Neural Network** demonstrated superior accuracy and robustness in noisy environments.
* **Latency**: Optimized the inference loop to achieve sub-100ms processing time, ensuring a seamless user experience for live command control.
