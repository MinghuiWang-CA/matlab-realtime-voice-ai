% =========================================================================
% Voice Command Classification - Real-Time Control System
% =========================================================================
% Interactive "Bubble Shooter" controlled by real-time voice inference.
% Targeted Commands: Up, Down, Left, Right (Mandarin Chinese)
% =========================================================================

clear all; close all; clc;

% System Configuration
fact = 4;           % Processing speed factor
fs = 44.1e3; 
Nb_ech_acq = fs;
win_limit = 10;     % Number of hits to win

% --- 1. Load & Initialize Feature Extractor ---
% [Insert afe configuration and normalization vectors here]

% --- 2. Real-Time Audio Stream Initialization ---
adr = audioDeviceReader('SampleRate', fs, 'SamplesPerFrame', Nb_ech_acq/fact);

% Game Variables
curseurx = 0; curseury = 0;
v_step = 0.4;
hitCount = 0;
switch_target = 1;
r = 0.5; th = 0:pi/50:2*pi;
xu = r*cos(th); yu = r*sin(th);

% --- 3. Main Control Loop ---
h = figure('Name', 'AI Voice Control - Bubble Shooter');
while hitCount < win_limit
    % Audio Acquisition & Filtering
    audio_in = adr();
    audio_f = filter(b, a, audio_in);
    
    % Real-Time Inference
    feats = extract(afe, audio_f)';
    v_norm = (reshape(feats, [], 1) - mean_norm) ./ var_norm;
    predicted_code = trainedKNN.predictFcn(v_norm');
    
    % Command Mapping (Class to Movement)
    switch predicted_code
        case 2, curseurx = curseurx + v_step; % Command: Right
        case 1, curseurx = curseurx - v_step; % Command: Left
        case 3, curseury = curseury + v_step; % Command: Up
        case 0, curseury = curseury - v_step; % Command: Down
    end
    
    % Target Management
    if switch_target
        angle_rand = randi(360);
        dist_rand = (rand < 0.5) * 2 + 2; % Position at 2 or 4 units
        xc = dist_rand * cosd(angle_rand);
        yc = dist_rand * sind(angle_rand);
        switch_target = 0;
    end
    
    % Collision Detection
    if inpolygon(curseurx, curseury, xu+xc, yu+yc)
        switch_target = 1;
        hitCount = hitCount + 1;
        fprintf('Target Hit! Total: %d/%d\n', hitCount, win_limit);
    end
    
    % Graphical Update
    cla;
    plot(xu+xc, yu+yc, 'r', 'LineWidth', 2); hold on;
    plot(curseurx, curseury, 'b+', 'MarkerSize', 15, 'LineWidth', 2);
    axis equal; xlim([-5 5]); ylim([-5 5]);
    grid on; title(['Score: ' num2str(hitCount)]);
    drawnow;
    
    if ~ishandle(h) || double(get(h,'CurrentCharacter')) == 27, break; end
end

if hitCount >= win_limit, disp('Mission Accomplished: 10/10 targets hit!'); end
release(adr);