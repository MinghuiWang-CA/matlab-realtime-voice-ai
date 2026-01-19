function [trainedClassifier, validationAccuracy] = KNN_10(trainingData)
% [trainedClassifier, validationAccuracy] = KNN_10(trainingData)
% Returns a trained classifier and its accuracy, adapted for fact=10.

% Auto-generated with dynamic VariableNames adjustment

% Extract predictors and response
% Convert input to table with dynamic variable names
nVars = size(trainingData,2);
varNames = arrayfun(@(k) sprintf('column_%d', k), 1:nVars, 'UniformOutput', false);
inputTable = array2table(trainingData, 'VariableNames', varNames);

% Define predictor and response columns
targetCol = sprintf('column_%d', nVars);
predictorNames = varNames(1:end-1);
predictors = inputTable(:, predictorNames);
response = inputTable.(targetCol);

% Train a Fine KNN classifier
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add classifier and metadata
testedVarCount = numel(predictorNames);
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'Exported KNN for fact=10';
trainedClassifier.HowToPredict = sprintf([
    'To make predictions, use: \n', ...
    '  yfit = trainedClassifier.predictFcn(X)\n', ...
    'where X is a matrix with %d columns corresponding to predictor features.'], testedVarCount);

% 5-fold cross-validation
partitionedModel = crossval(classificationKNN, 'KFold', 5);

% Compute validation predictions and accuracy
[~, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end
