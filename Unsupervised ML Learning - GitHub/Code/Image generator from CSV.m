%% Combined PRPD Dataset Expansion with Improved Augmentation & Density Normalization
% - Reads CSV files (with 'angle' and 'charge' columns).
% - Aggregates global density map for better visualization consistency.
% - Generates augmented PRPD plots with controlled perturbation and scaling.
% - Normalizes density maps separately for original and augmented data to ensure consistent colour mapping.

clc; clear; close all;
rng(42); % Set random seed for repeatability

%% Select Folder Containing CSV Files
folderPath = uigetdir(pwd, 'Select Folder Containing CSV Files');
if folderPath == 0
    error('No folder selected. Exiting.');
end

% Get a list of all CSV files in the folder
csvFiles = dir(fullfile(folderPath, '*.csv'));
fprintf('Processing %d CSV files...\n', length(csvFiles));

%% Step 2: Aggregate Global Data for Density Calculation
allPhases = [];
allCharges = [];
densityMap = containers.Map('KeyType', 'char', 'ValueType', 'double');

for fileIdx = 1:length(csvFiles)
    fileName = csvFiles(fileIdx).name;
    filePath = fullfile(folderPath, fileName);
    
    try
        data = readtable(filePath, 'PreserveVariableNames', true);
    catch
        fprintf('  Could not read file: %s. Skipping.\n', fileName);
        continue;
    end
    
    if all(ismember({'angle', 'charge'}, data.Properties.VariableNames))
        phase = round(data.angle, 1);
        charge = round(data.charge, -1);
        
        allPhases = [allPhases; phase];
        allCharges = [allCharges; charge];
        
        for i = 1:length(phase)
            key = sprintf('%0.1f_%d', phase(i), charge(i));
            if isKey(densityMap, key)
                densityMap(key) = densityMap(key) + 1;
            else
                densityMap(key) = 1;
            end
        end
    else
        fprintf('  Required columns missing in file: %s. Skipping.\n', fileName);
    end
end

if isempty(densityMap)
    error('No valid data found. Exiting.');
end

globalMaxDensity = max(cell2mat(values(densityMap)));
fprintf('Global maximum density found: %d\n', globalMaxDensity);

%% Augmentation Parameters
numAugmentedSamples = 10;  % Increased to improve density overlap
maxPhaseShift = 10;        % Max phase shift in degrees
chargeScalingRange = [0.97, 1.03];  % Slightly tighter scaling
noiseStdDev = 25;          % Reduced Gaussian noise

fprintf('Generating PRPD plots with improved augmentation...\n');

for fileIdx = 1:length(csvFiles)
    fileName = csvFiles(fileIdx).name;
    filePath = fullfile(folderPath, fileName);
    fprintf('Processing file: %s\n', fileName);
    
    try
        data = readtable(filePath, 'PreserveVariableNames', true);
    catch
        fprintf('  Could not read file: %s. Skipping.\n', fileName);
        continue;
    end
    
    if ~all(ismember({'angle', 'charge'}, data.Properties.VariableNames))
        fprintf('  Required columns missing in file: %s. Skipping.\n', fileName);
        continue;
    end
    
    %% 3A. Create Original PRPD Plot
    phaseOrig = round(data.angle, 1);
    chargeOrig = round(data.charge, -1);
    
    densityOrig = zeros(size(phaseOrig));
    for i = 1:length(phaseOrig)
        key = sprintf('%0.1f_%d', phaseOrig(i), chargeOrig(i));
        if isKey(densityMap, key)
            densityOrig(i) = densityMap(key);
        end
    end
    
    origMaxDensity = max(densityOrig);
    if origMaxDensity > 0
        densityOrig = densityOrig / origMaxDensity;  % Normalize within its own max density
    end
    
    createPRPDPlot(phaseOrig, chargeOrig, densityOrig, fileName, folderPath, 'Original');
    
    %% 3B. Create Augmented PRPD Plots
    for augIdx = 1:numAugmentedSamples
        augPhase = mod(data.angle + randi([-maxPhaseShift, maxPhaseShift], size(data.angle)), 360);
        
        scalingFactors = chargeScalingRange(1) + (chargeScalingRange(2) - chargeScalingRange(1)) * rand(size(data.charge));
        augCharge = data.charge .* scalingFactors + noiseStdDev * randn(size(data.charge));
        augCharge = max(augCharge, 0);
        
        augPhaseRounded = round(augPhase, 1);
        augChargeRounded = round(augCharge, -1);
        
        augDensity = zeros(size(augPhaseRounded));
        for i = 1:length(augPhaseRounded)
            key = sprintf('%0.1f_%d', augPhaseRounded(i), augChargeRounded(i));
            if isKey(densityMap, key)
                augDensity(i) = densityMap(key);
            end
        end
        
        augMaxDensity = max(augDensity);
        if augMaxDensity > 0
            augDensity = augDensity / augMaxDensity;  % Normalize within augmented max density
        end
        
        label = sprintf('Augmented_%d', augIdx);
        createPRPDPlot(augPhase, augCharge, augDensity, fileName, folderPath, label);
    end
end

fprintf('Processing complete. PRPD graphs saved in the same folder as the CSV files.\n');

function createPRPDPlot(phase, charge, density, fileName, folderPath, label)
    % Create figure
    figure('Units', 'Pixels', 'Position', [100, 100, 512, 512]); % Fixed size for CNN
    hold on;

    % Plot reference sine wave
    phaseAngles = linspace(0, 360, 1000);
    sineWave = 6000 * sin(deg2rad(phaseAngles));  
    plot(phaseAngles, sineWave, 'k--', 'LineWidth', 1);

    % Scatter plot with controlled marker transparency
    scatter(phase, charge, 30, density, 'filled', 'MarkerFaceAlpha', 0.9);
    colormap("parula");

    % Normalize color scale
    if max(density) > 0
        clim([0 max(density)]);
    else
        clim([0 1]); % Default scale when max density is zero
    end

    % Add axis labels and title
    xlabel('Phase Angle (degrees)', 'FontSize', 12);
    ylabel('Charge (pC)', 'FontSize', 12);
    
    % Show axes
    axis tight;
    box on; % Optional: adds a box around the plot
    set(gca, 'FontSize', 10);
    
    % Adjust layout for visibility
    set(gca, 'Position', [0.13 0.11 0.775 0.75]);  % Standard layout with margins

    % Save the figure
    outputFileName = sprintf('PRPD_%s_%s.png', fileName, label);
    outputFileName = strrep(outputFileName, ' ', '_');
    exportgraphics(gcf, fullfile(folderPath, outputFileName), 'Resolution', 300, 'BackgroundColor', 'white');

    close(gcf);
    hold off;
end
