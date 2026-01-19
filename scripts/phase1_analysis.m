% =========================================================================
% Voice Command Classification - Signal Processing & Feature Extraction
% =========================================================================
% This script performs:
% 1 - Audio data ingestion
% 2 - Digital filtering (Preprocessing)
% 3 - MFCC Feature extraction
% 4 - Data normalization and formatting for ML training
% =========================================================================

clear all; close all; clc;

% Configuration
fact = 1;
fs = 44.1e3; 
Nb_ech_acq = fs;

% 1 - Data Loading
chemin = fullfile(pwd, 'sons_audio');
ads = audioDatastore(chemin, 'IncludeSubfolders', true, ...
    'FileExtensions', '.wav', 'LabelSource', 'foldernames');

commands = categorical(["Son_a", "Son_i", "Son_o", "Son_ambiance"]);
isCommand = ismember(ads.Labels, commands);
adsTrain = subset(ads, isCommand);

% 2 - Preprocessing (Digital Filtering)
fc = 4000;        
Fn = fs / 2;      
Wn = fc / Fn;     
[b, a] = butter(4, Wn); % 4th order Butterworth filter

% 3 - Feature Extraction Configuration (MFCC)
Temps_Frame = 0.03;             
Saut_Temps = 0.02;             
Taille_FFT = 2048;           
Echantillon_Frame = round(Temps_Frame * fs);
Echantillon_Saut = round(Saut_Temps * fs);
Echantillon_Chevauche = Echantillon_Frame - Echantillon_Saut;

afe = audioFeatureExtractor( ...
    'SampleRate', fs, ...
    'FFTLength', Taille_FFT, ...
    'Window', hamming(Echantillon_Frame, 'periodic'), ...
    'OverlapLength', Echantillon_Chevauche, ...
    "mfcc", true);

% 4 - Feature Extraction Loop
XTrain = [];
x_all = [];
for ind = 1:numel(adsTrain.Files)
    x = read(adsTrain);
    x_all = [x_all; x];
    x_F = filter(b, a, x); % Noise reduction
    features = extract(afe, x_F)';
    XTrain(:, :, ind) = features;
end

% Visual Analysis (Temporal & Spectral)
t = (0:length(x_all) - 1) / fs;
figure('Name', 'Signal Analysis');
subplot(2,1,1);
plot(t, x_all);
title('Time Domain Signal');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,1,2);
N = length(x_all);
X = fft(x_all);
f = fs * (0:floor(N/2)) / N;
plot(f, abs(X(1:floor(N/2)+1)));
title('Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');

% 5 - Reshaping and Normalization
[nb_features, nb_fenetres, nb_indices] = size(XTrain);
XTrain_etal = reshape(XTrain, nb_features * nb_fenetres, nb_indices);

% Z-score Normalization
vect_mean = mean(XTrain_etal, 2);
vect_var  = std(XTrain_etal, 0, 2);
xTrain_Norm = (XTrain_etal - vect_mean) ./ vect_var;

% 6 - Label Processing
YTrain = removecats(adsTrain.Labels);
[Target, ~, ~] = grp2idx(YTrain);
Target = Target - 1;  

% Final Visualization of Normalized Features
figure('Name', 'Feature Visualization');
subplot(2,1,1);
pcolor(xTrain_Norm);
shading flat;
title('Normalized MFCC Features');
ylabel('Feature Index');

subplot(2,1,2);
plot(Target, 'LineWidth', 1.5);
title('Class Labels');
xlabel('Sample Index'); ylabel('Class');
grid on;