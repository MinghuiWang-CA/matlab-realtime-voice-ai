% =========================================================================
% Voice Command Classification - Machine Learning & Model Evaluation
% =========================================================================
% Comparison between K-Nearest Neighbors (KNN) and Neural Networks (NN).
% Includes data splitting (Train/Test) and accuracy reporting.
% =========================================================================

clear all; close all; clc;

% --- 1. Load Preprocessed Data & Extract Features ---
% (Assuming extraction logic from Phase 1 is executed here)
% [Insert your extraction and normalization code from Phase 1 here]

% --- 2. Dataset Preparation (Train/Test Split) ---
Data_ML = [xTrain_Norm; Target']';
Data_ML_APP = [];
Data_ML_DET = [];
training_ratio = 20; % 20% for training, 80% for testing
NB_classes = max(Data_ML(:,end)) + 1;

for n = 1 : NB_classes
    pos_son = find(Data_ML(:, end) == n-1);
    nb_exemples = length(pos_son);
    nb_app = floor(nb_exemples * training_ratio / 100);
    
    Data_ML_APP = [Data_ML_APP; Data_ML(pos_son(1 : nb_app), :)];
    Data_ML_DET = [Data_ML_DET; Data_ML(pos_son(nb_app+1 : end), :)];
end

% --- 3. K-Nearest Neighbors (KNN) Training ---
[trainedKNN, sons_estim_KNN_APP, valAccKnn] = KNN_1(Data_ML_APP);

% --- 4. Neural Network (Pattern Recognition) Training ---
X_app = Data_ML_APP(:, 1:end-1)';
Y_app = ind2vec(Data_ML_APP(:, end)' + 1, NB_classes);

net = patternnet(10); % 10 Hidden neurons
net.trainParam.showWindow = false;
net = train(net, X_app, Y_app);

% --- 5. Model Evaluation (Testing Phase) ---
X_test = Data_ML_DET(:, 1:end-1)';
Y_test = Data_ML_DET(:, end);

% KNN Prediction
sons_estim_KNN_DET = trainedKNN.predictFcn(X_test');

% NN Prediction
pred_RN_det = net(X_test);
[~, sons_estim_RN_DET] = max(pred_RN_det, [], 1);
sons_estim_RN_DET = sons_estim_RN_DET' - 1;

% --- 6. Results Visualization ---
figure('Name', 'Model Comparison: KNN vs Neural Network');
subplot(2,1,1);
plot(Y_test, 'k', 'LineWidth', 2); hold on;
plot(sons_estim_KNN_DET, 'r--', 'LineWidth', 1);
title('KNN Predictions vs Ground Truth');
legend('Target', 'KNN Prediction'); grid on;

subplot(2,1,2);
plot(Y_test, 'k', 'LineWidth', 2); hold on;
plot(sons_estim_RN_DET, 'b--', 'LineWidth', 1);
title('Neural Network Predictions vs Ground Truth');
legend('Target', 'NN Prediction'); grid on;

% Accuracy Reporting
accKNN = sum(sons_estim_KNN_DET == Y_test) / length(Y_test) * 100;
accRN  = sum(sons_estim_RN_DET == Y_test) / length(Y_test) * 100;
fprintf('Final Results:\n - KNN Accuracy: %.2f%%\n - NN Accuracy: %.2f%%\n', accKNN, accRN);