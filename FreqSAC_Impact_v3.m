%% SAC Modulation Spectrum Analysis System - Study on Multi-Parameter Impacts
% Study the impacts of parameters such as CM2, I_exc, I_hexc, f1, f2 on spectral components
clc; clear; close all;

%% Global Parameter Settings
globalParams = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'sigma_s', 2.7e-16, ...
    'lambda_s', 532e-7, ...
    'lambda_d', 561e-7, ...
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'sigma_d_ratio', 0.049850201 ...
);

%% Analysis Parameter Configuration
analysisConfig = struct(...
    'I_s', 10e3, ...                % Excitation light intensity (W/cm²)
    'I_d', 500e3, ...               % Competition light intensity (W/cm²)
    'f1', 10e3, ...                 % Excitation modulation frequency (Hz)
    'f2', 15e3, ...                 % Competition modulation frequency (Hz)
    'm_s', 1.0, ...                 % Excitation modulation contrast
    'm_d', 1.0, ...                 % Competition modulation contrast
    'duration', 1, ...              % Signal duration (s)
    'interval', 10e-6, ...          % Sampling interval (s)
    'high_res_interval', 0.1e-6 ... % High-resolution sampling interval
);

%% Auxiliary Function - Calculate SAC Signal and Spectral Components
function results = computeSACComponents(globalParams, I_s, I_d, f1, f2, m_s, m_d, duration, interval)
    % Calculate relevant constants
    h = globalParams.h;
    c = globalParams.c;
    c1 = 1 + globalParams.k_isc/globalParams.k_t;
    sigma_d = globalParams.sigma_s * globalParams.sigma_d_ratio;
    
    % Calculate excitation and competition rates
    k_s = globalParams.sigma_s * I_s * globalParams.lambda_s / (h * c);
    k_d = sigma_d * I_d * globalParams.lambda_d / (h * c);
    
    % Generate time series
    t = 0:interval:duration-interval;
    n = length(t);
    
    % Generate modulation signals
    cos_f1 = cos(2*pi*f1*t);
    cos_f2 = cos(2*pi*f2*t);
    
    numerator = k_s * (1 + m_s * cos_f1);
    denominator = c1*(k_s*(1 + m_s * cos_f1) + ...
                   k_d*(1 + m_d * cos_f2)) + globalParams.k0;
    y_s = numerator ./ denominator;
    
    % Fourier transform analysis
    f_fft = fft(y_s);
    f_fft_shift = fftshift(f_fft);
    result = abs(f_fft_shift) / max(abs(f_fft_shift));
    
    % Calculate frequency indices
    freq_resolution = (1/interval)/n;
    indices = round([f1, f2, f1+f2, abs(f2-f1), 2*f1, 3*f1] / freq_resolution) + n/2 + 1;
    
    % Ensure indices are within valid range
    indices = min(max(indices, 1), n);
    
    % Calculate frequency component ratios
    sumx = (sum(result) - result(n/2+1)) / 2; 
    if f1 == f2
        results = struct(...
            'fund', result(indices(1)) / sumx, ...
            'harm', result(indices(2)) / sumx, ...
            'sum', result(indices(3)) / sumx, ...
            'diff', result(indices(4)-1) / sumx, ...
            'double', result(indices(5)) / sumx, ...
            'triple', result(indices(6)) / sumx ...
        );
    else
        results = struct(...
            'fund', result(indices(1)) / sumx, ...
            'harm', result(indices(2)) / sumx, ...
            'sum', result(indices(3)) / sumx, ...
            'diff', result(indices(4)) / sumx, ...
            'double', result(indices(5)) / sumx, ...
            'triple', result(indices(6)) / sumx ...
        );
    end
end

%% Auxiliary Function - Create Professional Plots
function createProfessionalPlot(xData, yData, labels, xLabel, yLabel, titleText, legendText)
    figure('Position', [100, 100, 1000, 700], 'Color', 'w', 'Name', titleText);   
    colors = lines(size(yData, 2));
    lineStyles = {'-', '--', ':', '-.', '-', '--'};
    lineWidths = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    markers = {'o', 's', '^', 'd', 'v', '+'};
    hold on;
    for i = 1:size(yData, 2)
        if length(xData) <= 10 
            plot(xData, yData(:,i), ...
                'Color', colors(i,:), ...
                'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1}, ...
                'LineWidth', lineWidths(i), ...
                'Marker', markers{mod(i-1, length(markers)) + 1}, ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'DisplayName', labels{i});
        else
            plot(xData, yData(:,i), ...
                'Color', colors(i,:), ... 
                'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1}, ...
                'LineWidth', lineWidths(i), ...
                'DisplayName', labels{i});
        end
    end
    hold off;
    
    % Set plot properties
    grid on;
    % grid minor;
    ylim([0, 100]);
    set(gca, 'LineWidth', 2, 'FontSize', 18, 'FontWeight', 'bold', ...
             'XMinorTick', 'on', 'YMinorTick', 'on', ...
             'TickLength', [0.02, 0.02]);
    
    xlabel(xLabel, 'FontSize', 24, 'FontWeight', 'bold');
    ylabel(yLabel, 'FontSize', 24, 'FontWeight', 'bold');
    title(titleText, 'FontSize', 24, 'FontWeight', 'bold');
    
    legend('show', 'Location', 'northeast', 'FontSize', 12, 'Box', 'off');
    set(gcf, 'Color', 'w');
end

%% Progress Display Function
function showProgress(current, total, message)
    if mod(current, ceil(total/10)) == 0 || current == total
        fprintf('%s: %.0f%%\n', message, current/total*100);
    end
end

%% 1. Impact of CM1 Variation on Spectral Components
fprintf('=== Analysis 1: Impact of CM1 Modulation Depth on Spectral Components ===\n');
m_s_range = 0:0.01:1;
components_relative_1 = zeros(length(m_s_range), 6);

for i = 1:length(m_s_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,analysisConfig.I_d,...
        analysisConfig.f1,analysisConfig.f2,m_s_range(i),analysisConfig.m_d,analysisConfig.duration,analysisConfig.interval);
    
    components_relative_1(i,:) = [results.fund, results.sum, results.diff, ...
                               results.double, results.triple, results.harm];
    
    showProgress(i, length(m_s_range), 'CM1 Analysis Progress');
end

labels = {'ξ(f_1)', 'ξ(f_1+f_2)', 'ξ(f_1-f_2)', 'ξ(2f_1)', 'ξ(3f_1)', 'ξ(f_2)'};
createProfessionalPlot(m_s_range*100, components_relative_1*100, labels, ...
    'CM_1 Modulation Depth (%)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs CM_1 Modulation Depth', labels);

%% 2. Impact of CM2 Variation on Spectral Components
fprintf('\n=== Analysis 2: Impact of CM2 Modulation Depth on Spectral Components ===\n');
m_d_range = 0:0.01:1;
components_relative_2 = zeros(length(m_d_range), 6);

for i = 1:length(m_d_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,analysisConfig.I_d,...
        analysisConfig.f1,analysisConfig.f2,analysisConfig.m_s,m_d_range(i),analysisConfig.duration,analysisConfig.interval);
    
    components_relative_2(i,:) = [results.fund, results.sum, results.diff, ...
                               results.double, results.triple, results.harm];
    
    showProgress(i, length(m_d_range), 'CM2 Analysis Progress');
end

labels = {'ξ(f_1)', 'ξ(f_1+f_2)', 'ξ(f_1-f_2)', 'ξ(2f_1)', 'ξ(3f_1)', 'ξ(f_2)'};
createProfessionalPlot(m_d_range*100, components_relative_2*100, labels, ...
    'CM_2 Modulation Depth (%)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs CM_2 Modulation Depth', labels);

%% 3. Impact of Excitation Light Intensity (I_exc) Variation
fprintf('\n=== Analysis 3: Impact of Excitation Light Intensity on Spectral Components ===\n');
I_exc_range = linspace(0, 300, 1001)*1e3; 
components_exc = zeros(length(I_exc_range), 6);

for i = 1:length(I_exc_range)
    results = computeSACComponents(globalParams,I_exc_range(i), ...
        analysisConfig.I_d,analysisConfig.f1,analysisConfig.f2, ...
        analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
        analysisConfig.interval);
    
    components_exc(i,:) = [results.fund, results.sum, results.diff, ...
                          results.double, results.triple, results.harm];
    
    showProgress(i, length(I_exc_range), 'I_exc Analysis Progress');
end

createProfessionalPlot(I_exc_range/1e3, components_exc*100, labels, ...
    'I_{exc} (kW/cm^2)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Excitation Intensity', labels);

%% 4. Impact of Competition Light Intensity (I_hexc) Variation
fprintf('\n=== Analysis 4: Impact of Competition Light Intensity on Spectral Components ===\n');

I_hexc_range = linspace(0, 1000, 1001)*1e3; 
components_hexc = zeros(length(I_hexc_range), 6);

for i = 1:length(I_hexc_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,...
        I_hexc_range(i),analysisConfig.f1,analysisConfig.f2,...
        analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
        analysisConfig.interval);
    
    components_hexc(i,:) = [results.fund, results.sum, results.diff, ...
                          results.double, results.triple, results.harm];
    
    showProgress(i, length(I_hexc_range), 'I_hexc Analysis Progress');
end

createProfessionalPlot(I_hexc_range/1e6, components_hexc*100, labels, ...
    'I_{hexc} (MW/cm^2)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Competition Intensity', labels);

%% 5. Impact of Modulation Frequency (f1 and f2) Variation
fprintf('\n=== Analysis 5: Impact of Modulation Frequency on Spectral Components ===\n');

f1_range = linspace(1, 100, 100)*1e3;
components_f1 = zeros(length(f1_range), 6);

for i = 1:length(f1_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s, ...
        analysisConfig.I_d,f1_range(i),analysisConfig.f2,analysisConfig.m_s, ...
        analysisConfig.m_d,analysisConfig.duration,analysisConfig.high_res_interval);

    components_f1(i,:) = [results.fund, results.sum, results.diff, ...
        results.double, results.triple, results.harm];


    showProgress(i, length(f1_range), 'f1 Frequency Analysis Progress');
end

f2_range = linspace(1, 100, 100)*1e3; 
components_f2 = zeros(length(f2_range), 6);

for i = 1:length(f2_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s, ...
        analysisConfig.I_d,analysisConfig.f1,f2_range(i),analysisConfig.m_s, ...
        analysisConfig.m_d,analysisConfig.duration,analysisConfig.high_res_interval);
    
    components_f2(i,:) = [results.fund, results.sum, results.diff, ...
                         results.double, results.triple, results.harm];
    
    showProgress(i, length(f2_range), 'f2 Frequency Analysis Progress');
end

% Plot frequency impact results
createProfessionalPlot(f1_range/1e3, components_f1*100, labels, ...
    'f_1 (kHz)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Excitation Modulation Frequency f_1', labels);

createProfessionalPlot(f2_range/1e3, components_f2*100, labels, ...
    'f_2 (kHz)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Competition Modulation Frequency f_2', labels);

%% 6. Frequency Combination Optimization Analysis
fprintf('\n=== Frequency Combination Optimization Analysis ===\n');

f1_test_range = linspace(1, 100, 10)*1e3;
f2_test_range = linspace(1, 100, 10)*1e3;
optimization_matrix = zeros(length(f1_test_range), length(f2_test_range));

total_points = length(f1_test_range) * length(f2_test_range);
current_point = 0;

for i = 1:length(f1_test_range)
    for j = 1:length(f2_test_range)
        results = computeSACComponents(globalParams,analysisConfig.I_s, ...
            analysisConfig.I_d,f1_test_range(i),f2_test_range(j), ...
            analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
            analysisConfig.high_res_interval);
        
        optimization_matrix(i,j) = results.fund * results.harm * 10000;
        
        current_point = current_point + 1;
        showProgress(current_point, total_points, 'Frequency Combination Optimization Progress');
    end
end

% Plot optimization heatmap
figure('Position', [100, 100, 800, 800], 'Color', 'w');
imagesc(f2_test_range/1e3, f1_test_range/1e3, optimization_matrix);
colorbar;
set(gca, 'YDir', 'normal', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('f_2 (kHz)', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('f_1 (kHz)', 'FontSize', 24, 'FontWeight', 'bold');
title('Frequency Combination Optimization (ξ(f_1)×ξ(f_2))', 'FontSize',24, 'FontWeight', 'bold');
colormap(jet);
set(gcf, 'Color', 'w');

%% 7. Performance Metric Calculation and Result Saving
fprintf('\n=== Performance Metric Calculation and Result Saving ===\n');

% Calculate key performance metrics
performanceMetrics = struct(...
    'max_CM1_performance', max(components_relative_1(:)), ...
    'optimal_CM1', m_s_range(components_relative_1(:,1) == max(components_relative_1(:,1))), ...
    'max_CM2_performance', max(components_relative_2(:)), ...
    'optimal_CM2', m_d_range(components_relative_2(:,1) == max(components_relative_2(:,1))), ...
    'optimal_I_exc', I_exc_range(components_exc(:,1) == max(components_exc(:,1))) / 1e3, ...
    'optimal_I_hexc', I_hexc_range(components_hexc(:,1) == max(components_hexc(:,1))) / 1e6, ...
    'optimal_f1', f1_range(components_f1(:,1) == max(components_f1(:,1))) / 1e3, ...
    'optimal_f2', f2_range(components_f2(:,1) == max(components_f2(:,1))) / 1e3 ...
);

% Save results
analysisResults = struct(...
    'CM1_analysis', components_relative_1, ...
    'CM2_analysis', components_relative_2, ...
    'I_exc_analysis', components_exc, ...
    'I_hexc_analysis', components_hexc, ...
    'f1_analysis', components_f1, ... 
    'f2_analysis', components_f2, ...
    'frequency_optimization', optimization_matrix, ...
    'globalParams', globalParams, ...
    'analysisConfig', analysisConfig, ...
    'performanceMetrics', performanceMetrics, ...
    'timestamp', datetime('now') ...
);

% save('SAC_Analysis_Results.mat', 'analysisResults');
% fprintf('Analysis results saved to SAC_Analysis_Results.mat\n');

%% 8. Generate Summary Report
fprintf('\n===== SAC Modulation Spectrum Analysis Summary =====\n');
fprintf('Analysis completion time: %s\n', datestr(now));
fprintf('Total analysis data points: %d\n', length(m_s_range)+length(m_d_range) + ...
    length(I_exc_range) + length(I_hexc_range) + length(f1_range) + ...
    length(f2_range) + numel(optimization_matrix));
fprintf('\nKey Findings:\n');
fprintf('1. Optimal CM1 modulation depth: %.2f\n', performanceMetrics.optimal_CM1);
fprintf('2. Optimal CM2 modulation depth: %.2f\n', performanceMetrics.optimal_CM2);
fprintf('3. Optimal excitation light intensity: %.1f kW/cm²\n', performanceMetrics.optimal_I_exc);
fprintf('4. Optimal competition light intensity: %.1f MW/cm²\n', performanceMetrics.optimal_I_hexc); 
fprintf('5. Optimal modulation frequency f1: %.1f kHz\n', performanceMetrics.optimal_f1);
fprintf('6. Optimal modulation frequency f2: %.1f kHz\n', performanceMetrics.optimal_f2);
fprintf('7. Maximum performance metric: %.4f\n', performanceMetrics.max_CM2_performance);

fprintf('\nAnalysis completed! All results have been saved and visualized.\n');