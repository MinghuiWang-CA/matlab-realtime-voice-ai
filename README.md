# matlab-realtime-voice-ai

This project implements a complete **Voice Detection and Recognition System** using MATLAB. It features a full pipeline from digital signal preprocessing to real-time inference using Machine Learning (KNN) and Deep Learning (Neural Networks) to classify voice commands and control an interactive application.

<br>

## Project Evolution (3-Phase Pipeline)
The project was developed in three distinct stages to ensure a robust final system:

### Phase 1: Signal Analysis & Preprocessing
* **Filtering**: Implementation of Butterworth filters (4th order) to isolate voice frequencies.
* **Feature Extraction**: Used MFCC (Mel-Frequency Cepstral Coefficients) with Hamming windows to represent vocal characteristics efficiently.

### Phase 2: Model Training & Comparison
* **Architectures**: Comparative study between **K-Nearest Neighbors (KNN)** and **Pattern Recognition Neural Networks**.
* **Validation**: Extensive performance evaluation using cross-validation and confusion matrices to minimize overfitting and ensure reliability.

### Phase 3: Real-Time Interactive System
* **Inference**: Optimized high-efficiency inference loop using `audioDeviceReader`.
* **Application**: Integration of the AI engine into a **Bubble Shooter game**, where voice commands control the cursor movement in real-time.

<br>

## Key Features
* **Latency Optimization**: Used a segmentation factor (`fact = 4`) to achieve sub-100ms processing, ensuring seamless live control.
* **Robust Classification**: Comparative results showing superior accuracy of Neural Networks in noisy environments.

<br>

## Software Stack
* **MATLAB**: Main development environment.
* **Toolboxes**: 
    * Signal Processing Toolbox
    * Deep Learning & Statistics Toolbox
    * Audio Toolbox

<br>

## Repository Structure
* `/scripts`: Contains the code for the three main phases of the project.
* `/models`: KNN and Neural Network training functions (e.g., `KNN_1.m`, `KNN_4.m`).
* `/data`: Sample `.wav` audio files used for training and testing (Up, Down, Left, Right).
