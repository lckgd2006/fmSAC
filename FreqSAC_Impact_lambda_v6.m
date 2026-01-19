%% SAC Modulation Spectrum Analysis System - Study on the Impact of Multi-Dye Wavelength Parameters (Flexible Version)
% Can choose to generate 7 independent charts or 1 combined chart
clc; clear; close all;

addpath(genpath('CSV'));
tic;

%% Plot Mode Selection
% Set to 'individual' to generate 7 independent charts, set to 'combined' to generate 1 combined chart
plotMode = 'individual';       % Can be modified to 'combined' or 'individual'

%% Dye File List
dyeFiles = {
    '540-560orange.csv', 'Orange Dye (540-560)';
    '580-605red.csv', 'Red Dye (580-605)'; 
    '625-645crimson.csv', 'Crimson Dye (625-645)';
    '640-660deep red.csv', 'Deep Red Dye (640-660)';
    'Qdot 565.csv', 'Qdot 565';
    'Qdot 605.csv', 'Qdot 605';
    'Rhodamine 6G.csv', 'Rhodamine 6G'
};

numDyes = size(dyeFiles, 1);

%% Global Parameter Settings
globalParams = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'sigma_s_ref', 2.7e-16, ...     % Reference absorption cross-section (at 532nm)
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'm_s', 1 ...
);

%% Analysis Parameter Configuration
analysisConfig = struct(...
    'I_s', 10e3, ...            % Excitation light intensity (W/cm²)
    'I_d', 1000e3, ...          % Competition light intensity (W/cm²)
    'f1', 10e3, ...             % Excitation modulation frequency (Hz)
    'f2', 15e3, ...             % Competition modulation frequency (Hz)
    'm_d', 0.6, ...             % Competition modulation contrast
    'duration', 0.1, ...        % Signal duration (s)
    'interval', 10e-6, ...      % Sampling interval (s)
    'lambda_s_fixed', 532, ...  % Fixed excitation wavelength (nm)
    'lambda_d_fixed', 561 ...   % Fixed competition wavelength (nm)
);

%% Wavelength Scanning Range
lambda_range = 480:5:680;

%% Auxiliary Function - Read Dye Data
function [wavelengths, excitation, emission] = readDyeData(filename)
    fprintf('Reading file: %s\n', filename);
    try
        data = readmatrix(filename);
        if size(data, 2) >= 3
            wavelengths = data(:, 1);
            excitation = data(:, 2);
            emission = data(:, 3);
        else
            wavelengths = data(:, 1);
            excitation = data(:, 2);
            emission = zeros(size(excitation));
        end
        
        % Process negative values and outliers
        excitation(excitation < 0) = 0;
        emission(emission < 0) = 0;
        
        % Normalize excitation and emission spectra
        if max(excitation) > 0
            excitation = excitation / max(excitation);
        else
            excitation = zeros(size(excitation));
        end
        
        if max(emission) > 0
            emission = emission / max(emission);
        else
            emission = zeros(size(emission));
        end
        
        fprintf('Data range: %d-%d nm, Excitation spectrum max value: %.4f, Emission spectrum max value: %.4f\n', ...
            min(wavelengths), max(wavelengths), max(excitation), max(emission));
        
    catch ME
        fprintf('Error reading file: %s\n', ME.message);
        wavelengths = 400:800;
        excitation = zeros(size(wavelengths));
        emission = zeros(size(wavelengths));
    end
end

%% Auxiliary Function - Get Excitation Spectrum Value at Specific Wavelength
function exc_value = getExcitationValue(wavelength, wavelengths, excitation_spectrum)
    if isempty(wavelengths) || isempty(excitation_spectrum)
        exc_value = 0;
        return;
    end
    
    if wavelength < min(wavelengths) || wavelength > max(wavelengths)
        exc_value = 0;
    else
        % Interpolation calculation
        exc_value = interp1(wavelengths, excitation_spectrum, wavelength, 'linear', 0);
        if isnan(exc_value) || exc_value < 0 
            % Check if interpolation result is NaN (Not a Number) or negative value
            exc_value = 0;
        end
    end
end

%% Auxiliary Function - Calculate SAC Signal and Spectrum Components (Simplified Stable Version)
function results = computeSACComponents(globalParams, I_s, I_d, f1, f2, m_d, ...
                                       duration, interval, lambda_s_nm, lambda_d_nm, ...
                                       wavelengths, excitation_spectrum)
    try
        % Convert wavelength unit：nm -> cm
        lambda_s = lambda_s_nm * 1e-7;
        lambda_d = lambda_d_nm * 1e-7;
        
        % Calculate relevant constants
        h = globalParams.h;
        c = globalParams.c;
        c1 = 1 + globalParams.k_isc/globalParams.k_t;
        
        % Get excitation spectrum values
        exc_value_s = getExcitationValue(lambda_s_nm, wavelengths, excitation_spectrum);
        exc_value_d = getExcitationValue(lambda_d_nm, wavelengths, excitation_spectrum);
        
        % Calculate relative absorption cross-section
        ref_exc_value = max(excitation_spectrum);
        if ref_exc_value <= 0
            ref_exc_value = 1;
        end
        
        sigma_s = globalParams.sigma_s_ref * (exc_value_s / ref_exc_value);
        sigma_d = globalParams.sigma_s_ref * (exc_value_d / ref_exc_value);
        
        % Ensure minimum absorption cross-section
        min_sigma = globalParams.sigma_s_ref * 1e-6;
        sigma_s = max(sigma_s, min_sigma);
        sigma_d = max(sigma_d, min_sigma);
        
        % Calculate excitation and competition rates
        k_s = sigma_s * I_s * lambda_s / (h * c);
        k_d = sigma_d * I_d * lambda_d / (h * c);
        
        % Generate time series
        t = 0:interval:duration-interval;
        n = length(t);
        
        % Generate modulation signals
        cos_f1 = cos(2*pi*f1*t);
        cos_f2 = cos(2*pi*f2*t);
        
        numerator = k_s * (1 + globalParams.m_s * cos_f1);
        denominator = c1*(k_s*(1 + globalParams.m_s * cos_f1) + ...
                       k_d*(1 + m_d * cos_f2)) + globalParams.k0;
        
        % Avoid division by zero
        denominator(denominator == 0) = 1e-30;
        y_s = numerator ./ denominator;
        
        % Fourier transform analysis
        f_fft = fft(y_s);
        f_fft_shift = fftshift(f_fft);
        result = abs(f_fft_shift) / max(abs(f_fft_shift));
        
        % Calculate frequency indices
        freq_resolution = (1/interval)/n;
        target_freqs = [f1, f2, f1+f2, abs(f2-f1), 2*f1, 3*f1];
        indices = round(target_freqs / freq_resolution) + floor(n/2) + 1;
        indices = min(max(indices, 1), n);
        
        % Calculate frequency component proportion
        dc_index = floor(n/2) + 1;
        total_power = (sum(result) - result(dc_index))/2;
        
        if total_power > 0
            fund = result(indices(1)) / total_power;
            harm = result(indices(2)) / total_power;
            sum_freq = result(indices(3)) / total_power;
            diff_freq = result(indices(4)) / total_power;
            double_freq = result(indices(5)) / total_power;
            triple_freq = result(indices(6)) / total_power;
        else
            fund = 0.01; harm = 0.01; sum_freq = 0.01; 
            diff_freq = 0.01; double_freq = 0.01; triple_freq = 0.01;
        end
        
        results = struct(...
            'fund', fund, 'harm', harm, 'sum', sum_freq, ...
            'diff', diff_freq, 'double', double_freq, 'triple', triple_freq, ...
            'sigma_s', sigma_s, 'sigma_d', sigma_d, 'success', true);
        
    catch
        % Return default values if calculation fails
        results = struct(...
            'fund', 0.01, 'harm', 0.01, 'sum', 0.01, ...
            'diff', 0.01, 'double', 0.01, 'triple', 0.01, ...
            'sigma_s', globalParams.sigma_s_ref * 1e-6, ...
            'sigma_d', globalParams.sigma_s_ref * 1e-6, ...
            'success', false);
    end
end

%% Auxiliary Function - Create Independent Plot for Single Dye
function createIndividualPlot(dyeName, lambda_range, components_s, components_d, ...
                             wavelengths, excitation, emission, labels, colors, lineStyles, lineWidths, dyeIdx)
    
    fig = figure('Position', [50, 100, 1800, 450], 'Color', 'w', ...
                'Name', sprintf('%s - SAC Wavelength Impact Analysis', dyeName));
    
    % 1st Subplot: Excitation Wavelength Impact
    subplot(1, 3, 1);
    ax1 = gca;
    % Position = [left, bottom, width, height] (normalized coordinates, range 0-1)
    ax1.Position = [0.04 0.12 0.28 0.7];
    hold on;
    for i = 1:6
        plot(lambda_range, components_s(:, i), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, ...
            'LineWidth', lineWidths(i), 'DisplayName', labels{i});
    end
    hold off;
    grid on;
    set(gca, 'FontSize', 9, 'FontWeight', 'bold');
    xlabel('Excitation Wavelength λ_s (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Signal Intensity Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s\nFrequency Components vs Excitation Wavelength λ_s', ...
        dyeName), 'FontSize', 12, 'FontWeight', 'bold');
    xlim([450, 700]);
    ylim([0, 100]);
    legend('show','Location','northeast','FontSize',6,'Box','off');
    
    % 2nd Subplot: Competition Wavelength Impact
    subplot(1, 3, 2);
    ax2 = gca;
    % Position = [left, bottom, width, height] (normalized coordinates, range 0-1)
    ax2.Position = [0.36 0.12 0.28 0.7];
    hold on;
    for i = 1:6
        plot(lambda_range, components_d(:, i), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, ...
            'LineWidth', lineWidths(i), 'DisplayName', labels{i});
    end
    hold off;
    grid on;
    set(gca, 'FontSize', 9, 'FontWeight', 'bold');
    xlabel('Competition Wavelength λ_d (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Signal Intensity Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s\nFrequency Components vs Competition Wavelength λ_d', ...
        dyeName), 'FontSize', 12, 'FontWeight', 'bold');
    xlim([450, 700]);
    ylim([0, 100]);
    legend('show','Location','northeast','FontSize',6,'Box','off');
        
    % 3rd Subplot: Excitation and Emission Spectra
    subplot(1, 3, 3);
    ax3 = gca;
    ax3.Position = [0.68 0.12 0.28 0.7];        
    % Determine valid wavelength range
    if ~isempty(wavelengths)
        valid_idx = wavelengths >= 400 & wavelengths <= 800;
        if sum(valid_idx) > 10  % At least 10 valid data points
            wavelengths_plot = wavelengths(valid_idx);
            excitation_plot = excitation(valid_idx);
            emission_plot = emission(valid_idx);

            grid on;
            set(gca, 'FontSize', 9, 'FontWeight', 'bold');
            yyaxis left
            plot(wavelengths_plot, excitation_plot, 'b-', 'LineWidth', 2, 'DisplayName', 'Excitation Spectrum');
            ylabel('Normalized Excitation Intensity (a.u.)', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');
            ylim([0, 1]);

            yyaxis right
            plot(wavelengths_plot, emission_plot, 'r-', 'LineWidth', 2, 'DisplayName', 'Emission Spectrum');
            ylabel('Normalized Emission Intensity (a.u.)', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
            ylim([0, 1]);

            xlabel('Wavelength (nm)', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('%s\n Excitation / Emission Spectrum', dyeName), 'FontSize', 12, 'FontWeight', 'bold');
            xlim([400, 800]);
            legend({'Excitation Spectrum','Emission Spectrum'},'Location','northeast','FontSize',6,'Box','off');
        else
            % Insufficient data, display prompt
            text(0.5, 0.5, 'Insufficient spectral data', 'HorizontalAlignment', 'center', ...
                'FontSize', 16, 'FontWeight', 'bold');
            xlim([0, 1]);
            ylim([0, 1]);
            set(gca, 'XTick', [], 'YTick', []);
            title(sprintf('%s\nSpectral Data', dyeName), 'FontSize', 14, 'FontWeight', 'bold');
        end
    else
        % No data case
        text(0.5, 0.5, 'No spectral data', 'HorizontalAlignment', 'center', ...
            'FontSize', 15, 'FontWeight', 'bold');
        xlim([0, 1]);
        ylim([0, 1]);
        set(gca, 'XTick', [], 'YTick', []);
        title(sprintf('%s\nSpectral Data', dyeName), 'FontSize', 14, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
 
    % Adjust subplot spacing
    sgtitle(sprintf('Dye: %s - fmSAC Modulation Spectrum Analysis', dyeName), ...
           'FontSize', 15, 'FontWeight', 'bold', 'Color', 'blue');
    
    % Save chart (optional)
    filename = sprintf('SAC_Analysis_%s.png', regexprep(dyeName, '[^a-zA-Z0-9]', '_'));
    saveas(fig, filename);
    fprintf('Chart saved: %s\n', filename);
end

%% Auxiliary Function - Create Combined Plot
function createCombinedPlot(allResults, labels, colors, lineStyles, lineWidths)
    % Create large figure
    figure('Position', [50, 50, 1800, 2500], 'Color', 'w', 'Name', 'Multi-Dye SAC Wavelength Impact Analysis');
    
    for dyeIdx = 1:length(allResults)
        results = allResults{dyeIdx};
        
        % 1st Column: Excitation Wavelength Impact
        subplot(7, 3, (dyeIdx-1)*3 + 1);
        hold on;
        for i = 1:6
            plot(results.lambda_s_analysis.wavelengths, results.lambda_s_analysis.components(:,i), ...
                'Color', colors(i,:), 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i));
        end
        hold off;
        grid on;
        ylabel('Signal Intensity Ratio (%)', 'FontSize', 9, 'FontWeight', 'bold');
        if dyeIdx == 7
            xlabel('Excitation Wavelength λ_s (nm)', 'FontSize', 10, 'FontWeight', 'bold');
        end
        title(sprintf('%s\nExcitation Wavelength Impact', results.name), 'FontSize', 10, 'FontWeight', 'bold');
        ylim([0, 100]);
        
        % 2nd Column: Competition Wavelength Impact
        subplot(7, 3, (dyeIdx-1)*3 + 2);
        hold on;
        for i = 1:6
            plot(results.lambda_d_analysis.wavelengths, results.lambda_d_analysis.components(:,i), ...
                'Color', colors(i,:), 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i));
        end
        hold off;
        grid on;
        if dyeIdx == 7
            xlabel('Competition Wavelength λ_d (nm)', 'FontSize', 10, 'FontWeight', 'bold');
        end
        title(sprintf('%s\nCompetition Wavelength Impact', results.name), 'FontSize', 10, 'FontWeight', 'bold');
        ylim([0, 100]);
        
        % 3rd Column: Excitation and Emission Spectra
        subplot(7, 3, (dyeIdx-1)*3 + 3);
        wavelengths_spectra = results.spectra.wavelengths;
        excitation_spectra = results.spectra.excitation;
        emission_spectra = results.spectra.emission;
        
        % Find valid data range
        valid_idx = wavelengths_spectra >= 400 & wavelengths_spectra <= 800;
        if sum(valid_idx) > 0
            wavelengths_plot = wavelengths_spectra(valid_idx);
            excitation_plot = excitation_spectra(valid_idx);
            emission_plot = emission_spectra(valid_idx);
            
            yyaxis left
            plot(wavelengths_plot, excitation_plot, 'b-', 'LineWidth', 2);
            ylabel('Excitation Spectrum', 'FontSize', 9, 'Color', 'b', 'FontWeight', 'bold');
            ylim([0, 1]);
            
            yyaxis right
            plot(wavelengths_plot, emission_plot, 'r-', 'LineWidth', 2);
            ylabel('Emission Spectrum', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');
            ylim([0, 1]);
            
            xlabel('Wavelength (nm)', 'FontSize', 10, 'FontWeight', 'bold');
            title(sprintf('%s\nExcitation/Emission Spectrum', results.name), 'FontSize', 10, 'FontWeight', 'bold');
            grid on;
            xlim([450, 700]);
        else
            % If no valid data, display blank plot
            text(0.5, 0.5, 'No spectral data', 'HorizontalAlignment', 'center', 'FontSize', 12);
            xlim([0, 1]);
            ylim([0, 1]);
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
end

%% Main Analysis Loop
fprintf('Starting analysis of %d dyes...\n', numDyes);
fprintf('Plot mode: %s\n', plotMode);

% Spectrum component labels and styles
labels = {'ξ(f_1)', 'ξ(f_2)', 'ξ(f_1+f_2)', 'ξ(|f_1-f_2|)', 'ξ(2f_1)', 'ξ(3f_1)'};
colors = lines(6);
lineStyles = {'-', '--', ':', '-.', '-', '--'};
lineWidths = [2.5, 2.0, 2.0, 1.8, 1.8, 1.8];

% Store all results
allResults = cell(numDyes, 1);

for dyeIdx = 1:numDyes
    fprintf('\n=== Analyzing dye %d/%d: %s ===\n', dyeIdx, numDyes, dyeFiles{dyeIdx, 2});
    
    % Read dye data
    [wavelengths, excitation, emission] = readDyeData(dyeFiles{dyeIdx, 1});
    
    % Analysis 1: lambda_s scan, lambda_d fixed
    fprintf('Performing excitation wavelength scan analysis...\n');
    components_lambda_s = zeros(length(lambda_range), 6);
    
    for i = 1:length(lambda_range)
        results = computeSACComponents(globalParams, analysisConfig.I_s, analysisConfig.I_d, ...
            analysisConfig.f1, analysisConfig.f2, analysisConfig.m_d, analysisConfig.duration, ...
            analysisConfig.interval, lambda_range(i), analysisConfig.lambda_d_fixed, ...
            wavelengths, excitation);
        
        components_lambda_s(i,:) = [results.fund, results.harm, results.sum, ...
                                   results.diff, results.double, results.triple];
        
        if mod(i, 20) == 0
            fprintf('Progress: %d/%d\n', i, length(lambda_range));
        end
    end
    
    % Analysis 2: lambda_s fixed, lambda_d scan
    fprintf('Performing competition wavelength scan analysis...\n');
    components_lambda_d = zeros(length(lambda_range), 6);
    
    for i = 1:length(lambda_range)
        results = computeSACComponents(globalParams, analysisConfig.I_s, analysisConfig.I_d, ...
            analysisConfig.f1, analysisConfig.f2, analysisConfig.m_d, analysisConfig.duration, ...
            analysisConfig.interval, analysisConfig.lambda_s_fixed, lambda_range(i), ...
            wavelengths, excitation);
        
        components_lambda_d(i,:) = [results.fund, results.harm, results.sum, ...
                                   results.diff, results.double, results.triple];
        
        if mod(i, 20) == 0
            fprintf('Progress: %d/%d\n', i, length(lambda_range));
        end
    end
    
    % Store results (for combined plot)
    allResults{dyeIdx} = struct(...
        'name', dyeFiles{dyeIdx, 2}, ...
        'spectra', struct('wavelengths', wavelengths, 'excitation', excitation, 'emission', emission), ...
        'lambda_s_analysis', struct('wavelengths', lambda_range, 'components', components_lambda_s * 100), ...
        'lambda_d_analysis', struct('wavelengths', lambda_range, 'components', components_lambda_d * 100) ...
    );
    
    % Select plotting method according to mode
    if strcmp(plotMode, 'individual')
        % Create independent chart for current dye
        createIndividualPlot(dyeFiles{dyeIdx, 2}, lambda_range, ...
            components_lambda_s * 100, components_lambda_d * 100, ...
            wavelengths, excitation, emission, labels, colors, lineStyles, lineWidths, dyeIdx);
    end
    
    fprintf('Completed analysis of %s\n', dyeFiles{dyeIdx, 2});
end

% If combined mode is selected, create combined chart
if strcmp(plotMode, 'combined')
    fprintf('\n=== Generating combined plot ===\n');
    createCombinedPlot(allResults, labels, colors, lineStyles, lineWidths);
end

%% Save Analysis Results (optional)
% analysisResults = struct(...
%     'allResults', {allResults}, ...
%     'globalParams', globalParams, ...
%     'analysisConfig', analysisConfig, ...
%     'dyeFiles', {dyeFiles}, ...
%     'timestamp', datetime('now') ...
% );
% 
% save('MultiDye_SAC_Wavelength_Analysis.mat', 'analysisResults');
% fprintf('Analysis results saved to MultiDye_SAC_Wavelength_Analysis.mat\n');

fprintf('\n===== Analysis Completed =====\n');
elapsedTime = toc;
fprintf('Total runtime: %.2f seconds\n', elapsedTime);