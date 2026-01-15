%% fmSAC Alpha Coefficient Optimization and Performance Evaluation
% This script optimizes the alpha coefficient (α) for frequency-modulated Saturated Absorption Competition (fmSAC) microscopy.
% It evaluates the impact of α on imaging performance metrics (FWHM and negative intensity components)
% and compares three comprehensive performance evaluation methods to determine the optimal α value.

clc
clear all
close all
addpath(genpath('PSF'));
addpath(genpath('CSV'));
tic;% Start timing

%% Parameter Initialization
k_isc = 1.1e6;             % Intersystem crossing rate (1/s)
k_t = 0.49e6;              % Triplet state decay rate (1/s)
k0 = 2.56e8;               % Fluorescence decay rate (1/s)
c1 = 1 + k_isc/k_t;        % Precomputed constant for rate equation
h = 6.626e-34;             % Planck constant (J·s)
c = 3e10;                  % Speed of light (cm/s)
lambda_s = 532e-7;         % Excitation light wavelength (cm)
lambda_d = 488e-7;         % Competition light wavelength (cm)
sigma_s = 2.7e-16;         % Excitation light absorption cross-section (cm²)
sigma_d = sigma_s * 0.512063188; % Competition light absorption cross-section (cm²)
I_s = 10e3;                % Excitation light intensity (10 kW/cm²)
I_d = 500e3;               % Competition light intensity (500 kW/cm²)
f1 = 10e3;                 % Excitation modulation frequency f1 (10 kHz)
f2 = 15e3;                 % Competition modulation frequency f2 (15 kHz)
interval = 10e-6;          % Sampling interval
t = 0:interval:1-interval; % Precomputed time sequence
m_s = 1;                   % Excitation modulation contrast CM1
m_d = 0.9;                 % Competition modulation contrast CM2

%% Load 2D PSF Data
a=load('I_exc532_51_3D.mat');  % Excitation light PSF file (532nm, Gaussian)
I1=a.result.PSF(:,:,25);
b=load('I_hexc488_51_3D.mat'); % Competition light PSF file (488nm, doughnut)
I2=b.result.PSF(:,:,25);

% Normalization and Scaling of Light Intensity
I1 = I1 / max(I1(:));      % Normalize excitation PSF to [0,1]
I2 = I2 / max(I2(:));      % Normalize competition PSF to [0,1]
I_exc = I_s * I1;   
I_hexc = I_d * I2;

% Get PSF Dimensions
[rows, cols] = size(I1);
% Retrieve the center column
center_col = round(cols/2); 

%% Define relevant parameters
% Define Alpha Coefficient Range
alpha_coeffs = 0:0.1:2.5;                 % Range of alpha coefficients
num_coeffs = length(alpha_coeffs);

% Initialize Result Arrays
FWHM = zeros(1, num_coeffs);              % Full Width at Half Maximum (nm)
Neg_vals = zeros(1, num_coeffs);          % Negative intensity component (a.u.)
fmSAC_profiles = zeros(rows, num_coeffs); % fmSAC profiles (a.u.)

% FFT-related parameters
N = length(t);
frequencies = (-N/2:N/2-1) * (1/(N*interval));
f1_idx = find(abs(frequencies - f1) == min(abs(frequencies - f1)), 1);
f2_idx = find(abs(frequencies - f2) == min(abs(frequencies - f2)), 1);

%% Calculate the FWHM of fmSAC
% Create Progress Bar
fprintf('Calculate the FWHM of fmSAC...\n');
progressBar = waitbar(0, 'Calculating progress: 0%', 'Name', 'α coefficient scanning');

% Alpha coefficient optimization and performance Evaluation
for m = 1:num_coeffs
    current_alpha = alpha_coeffs(m); 
    sig_fund = zeros(rows, 1);  % fundamental frequency components
    sig_harm = zeros(rows, 1);  % harmonic frequency components
    
    for i = 1:rows
        % Calculate rate constant
        k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
        k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
        
        % Generate modulated signal
        y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
              (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);
        
        % FFT spectrum results
        f_omiga = fft(y_s);
        f_omiga_shift = fftshift(f_omiga);
        result = abs(f_omiga_shift) / max(abs(f_omiga_shift));
        
        % Calculate sum of frequency components (excluding DC component)
        total_power = (sum(result) - result(N/2+1))/2;
        
        % Extract fundamental and harmonic frequency components
        sig_fund(i) = result(f1_idx) / total_power;
        sig_harm(i) = result(f2_idx) / total_power;
    end
    
    % Calculate the fmSAC signal
    fmSAC_signal = sig_fund - current_alpha * sig_harm;
    fmSAC_profiles(:, m) = fmSAC_signal;
    
    % Normalize to [0,1]
    fmSAC_signal = fmSAC_signal / max(fmSAC_signal);
    
    % Calculate FWHM
    half_max = max(fmSAC_signal) / 2;
    half_index = find(fmSAC_signal >= half_max);
    FWHM(m) = length(half_index);
    
    % Calculate negative intensity component
    Neg_vals(m) = min(fmSAC_signal);
    
    % Update progress bar
    waitbar(m/num_coeffs, progressBar, sprintf('Calculating progress: %.0f%% (α=%.1f)', m/num_coeffs*100, current_alpha));
end
close(progressBar);

%% Calculate comprehensive performance metrics for 3 methods
% Normalize FWHM and negative intensity component
normalized_FWHM = (FWHM - min(FWHM)) / (max(FWHM) - min(FWHM));
normalized_Neg = (abs(Neg_vals) - min(abs(Neg_vals))) / (max(abs(Neg_vals)) - min(abs(Neg_vals)));

% Method 1: Normalized weighted sum
weight_FWHM = 0.7;      % Weight of FWHM
weight_Neg = 0.3;       % Weight of negative component
performance_metric1 = weight_FWHM * normalized_FWHM + weight_Neg * normalized_Neg;
[best_performance1, optimal_idx1] = min(performance_metric1);
optimal_alpha1 = alpha_coeffs(optimal_idx1);

% Method 2: Geometric mean
performance_metric2 = sqrt(normalized_FWHM .* normalized_Neg);
[best_performance2, optimal_idx2] = min(performance_metric2);
optimal_alpha2 = alpha_coeffs(optimal_idx2);

% Method 3: Metric with penalty term
penalty = 1 + 0.5 * (abs(Neg_vals) > 0.1);         % penalty term
performance_metric3 = normalized_FWHM .* penalty;
[best_performance3, optimal_idx3] = min(performance_metric3);
optimal_alpha3 = alpha_coeffs(optimal_idx3);

% Find optimal alpha for each method 
optimal_results = [
    optimal_alpha1, FWHM(optimal_idx1), Neg_vals(optimal_idx1), best_performance1;
    optimal_alpha2, FWHM(optimal_idx2), Neg_vals(optimal_idx2), best_performance2;
    optimal_alpha3, FWHM(optimal_idx3), Neg_vals(optimal_idx3), best_performance3
];

%% Generate Visualization Plots
% figure 1
% Top-left: fmSAC Profile
fig=figure('Position', [50, 50, 1500, 950], 'Color', 'w', 'Name', 'The Effect of α Coefficient on fmSAC');
cmap = jet(num_coeffs);
subplot(2,2,1);
hold on;
for m = 1:num_coeffs
    normalized_profile = fmSAC_profiles(:, m) / max(fmSAC_profiles(:, m));
    if m ~= optimal_idx1 && m ~= optimal_idx2 && m ~= optimal_idx3
        plot(0:(rows-1), normalized_profile, 'LineWidth', 1, 'Color', [0.7, 0.7, 0.7], ...
            'HandleVisibility', 'off'); 
    end
end
% Plot the optimal lines of three methods
optimal_profile1 = fmSAC_profiles(:, optimal_idx1) / max(fmSAC_profiles(:, optimal_idx1));
optimal_profile2 = fmSAC_profiles(:, optimal_idx2) / max(fmSAC_profiles(:, optimal_idx2));
optimal_profile3 = fmSAC_profiles(:, optimal_idx3) / max(fmSAC_profiles(:, optimal_idx3));
h_opt1 = plot(0:(rows-1), optimal_profile1, 'LineWidth', 2, 'Color', [1, 0.2, 0.2], ...
    'DisplayName', sprintf('Method 1: α = %.1f', optimal_alpha1));
h_opt2 = plot(0:(rows-1), optimal_profile2, 'LineWidth', 2, 'Color', [0.2, 0.6, 1], ...
    'DisplayName', sprintf('Method 2: α = %.1f', optimal_alpha2));
h_opt3 = plot(0:(rows-1), optimal_profile3, 'LineWidth', 2, 'Color', [0.3, 0.8, 0.3], ...
    'DisplayName', sprintf('Method 3: α = %.1f', optimal_alpha3));
hold off;
grid on;
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('Position (nm)', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Normalized Intensity (a.u.)', 'FontWeight', 'bold', 'FontSize', 15);
title('fmSAC Profile as a Function of the α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
xlim([0, 50]);
ylim([-0.3, 1]);
legend('show', 'Location', 'northeast','FontSize', 9, 'Box','off');

% figure 2
% Top-right: FWHM vs Alpha
subplot(2, 2, 2);
h1 = scatter(alpha_coeffs, FWHM, 'filled', 'MarkerEdgeColor', [0.2, 0.6, 1], ...
    'DisplayName', 'FWHM data points');
hold on;
h2 = plot(alpha_coeffs, FWHM, 'Color', [0, 0, 0], 'LineWidth', 2, ...
    'DisplayName', 'FWHM Trendline'); 
% Mark the optimal points for the three methods
h3 = plot(optimal_alpha1, FWHM(optimal_idx1), 's', 'MarkerSize', 8, 'MarkerFaceColor', [1, 0.2, 0.2], ...
    'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Optimal α Coefficient in Method 1'));
h4 = plot(optimal_alpha2, FWHM(optimal_idx2), '+', 'MarkerSize', 8, 'MarkerFaceColor', [0.2, 0.6, 1], ...
    'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Optimal α Coefficient in Method 2'));
h5 = plot(optimal_alpha3, FWHM(optimal_idx3), '^', 'MarkerSize', 8, 'MarkerFaceColor', [0.3, 0.8, 0.3], ...
    'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Optimal α Coefficient in Method 3'));
% colormap(jet);
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α coefficient', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('FWHM (nm)', 'FontWeight', 'bold', 'FontSize', 15);
title('FWHM as a Function of α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
grid on;
xlim([0, 2.5]);
ylim([22, 34]);
legend([h1, h2, h3, h4, h5], 'Location', 'northeast', 'FontSize', 9, 'Box','off'); 

% figure 3
% Bottom-left: 3D waterfall plot of fmSAC profiles
subplot(2, 2, 3);
X = 0:(rows-1);
Y = alpha_coeffs;
Z = fmSAC_profiles'./max(fmSAC_profiles(:));
mesh(X, Y, Z);
colormap("winter");
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('Position (nm)', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('α coefficient', 'FontWeight', 'bold', 'FontSize', 15);
zlabel('Normalized Intensity (a.u.)', 'FontWeight', 'bold', 'FontSize', 15);
title('3D: fmSAC Profile Distribution', 'FontWeight', 'bold', 'FontSize', 15);
view(30, 30);
xlim([0, 50]);
ylim([0, 2.5]);
zlim([-0.3, 1]);
grid on;

% figure 4
% Bottom-right: Negative component vs Alpha coefficient
subplot(2, 2, 4);
area_handle = area(alpha_coeffs, Neg_vals, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', [0.6, 0.1, 0.1], 'LineWidth', 2);
set(area_handle, 'FaceAlpha', 0.6);
hold on;
plot(alpha_coeffs, zeros(size(alpha_coeffs)), 'k--', 'LineWidth', 2);
% Mark the optimal points for the three methods
plot(optimal_alpha1, Neg_vals(optimal_idx1), 's', 'MarkerSize', 8, 'MarkerFaceColor', [1, 0.2, 0.2], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Optimal α Coefficient in Method 1');
plot(optimal_alpha2, Neg_vals(optimal_idx2), '+', 'MarkerSize', 8, 'MarkerFaceColor', [0.2, 0.6, 1], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Optimal α Coefficient in Method 2');
plot(optimal_alpha3, Neg_vals(optimal_idx3), '^', 'MarkerSize', 8, 'MarkerFaceColor', [0.3, 0.8, 0.3], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Optimal α Coefficient in Method 3');
set(gca, 'LineWidth', 1.5, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α coefficient', 'FontWeight', 'bold', 'FontSize', 15);    
ylabel('Negative Value Intensity (a.u.)', 'FontWeight', 'bold', 'FontSize', 15);
title('Negative Component variation with α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
grid on   
legend('show', 'Location','northeast', 'FontSize', 9, 'Box','off');

% Add overall title
sgtitle(sprintf('The Effect of the α Coefficient on fmSAC Performance (CM₂=%.1f)', m_d), ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.4]);

%% Create a comparison chart of the three evaluation methods
figure('Position', [100, 100, 1500, 500], 'Color', 'w', 'Name', 'Comparison of Three Evaluation Methods');

% Subfigure 1: method 1 - Normalized Weighted Sum
subplot(1, 3, 1);
plot(alpha_coeffs, performance_metric1, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
hold on;
plot(optimal_alpha1, best_performance1, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Comprehensive Performance Metrics', 'FontWeight', 'bold', 'FontSize', 15);
title({'Method 1: Normalized Weighted Sum', sprintf(['Weight: FWHM = %.1f, ' ...
    'Negative Value = %.1f'], weight_FWHM, weight_Neg)}, 'FontWeight', 'bold', 'FontSize', 15);
grid on;
text(0.55, 0.95, sprintf('Optimal α = %.1f\nMetrics = %.3f\nFWHM = %d nm\nNegative Value = %.3f', ...
    optimal_alpha1, best_performance1, FWHM(optimal_idx1), Neg_vals(optimal_idx1)), ...
    'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 9, ...
    'BackgroundColor', 'NONE', 'VerticalAlignment', 'top');

% Subfigure 2: method 2 - Geometric Mean
subplot(1, 3, 2);
plot(alpha_coeffs, performance_metric2, 'b-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;
plot(optimal_alpha2, best_performance2, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Comprehensive Performance Metrics', 'FontWeight', 'bold', 'FontSize', 15);
title('Method 2: Geometric Mean', 'FontWeight', 'bold', 'FontSize', 15);
grid on;
text(0.55, 0.95, sprintf('Optimal α = %.1f\nMetrics = %.3f\nFWHM = %d nm\nNegative Value = %.3f', ...
    optimal_alpha2, best_performance2, FWHM(optimal_idx2), Neg_vals(optimal_idx2)), ...
    'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 9, ...
    'BackgroundColor', 'NONE', 'VerticalAlignment', 'top');

% Subfigure 3: method 3 - Metrics with Penalty Clauses
subplot(1, 3, 3);
plot(alpha_coeffs, performance_metric3, 'g-*', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'g');
hold on;
plot(optimal_alpha3, best_performance3, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Comprehensive Performance Metrics', 'FontWeight', 'bold', 'FontSize', 15);
title('Method 3: Metrics with Penalty Clauses', 'FontWeight', 'bold', 'FontSize', 15);
grid on;
text(0.55, 0.95, sprintf('Optimal α = %.1f\nMetrics = %.3f\nFWHM = %d nm\nNegative Value = %.3f', ...
    optimal_alpha3, best_performance3, FWHM(optimal_idx3), Neg_vals(optimal_idx3)), ...
    'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 9, ...
    'BackgroundColor', 'NONE', 'VerticalAlignment', 'top');
text(0.2, 0.7, 'Penalty Conditions: |Negative Value|>0.1', 'Units', ...
    'normalized', 'FontWeight', 'bold', 'FontSize', 9, ...
    'BackgroundColor', 'yellow' );

% Add overall title
sgtitle('Comparison of Three Evaluation Methods', 'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.4]);

%% Output Detailed Result Analysis
fprintf('\n=== α Coefficient Optimization Result Analysis ===\n');
fprintf('FWHM range: [%d, %d] nm\n', min(FWHM), max(FWHM));
fprintf('Negative value range: [%.3f, %.3f]\n', min(Neg_vals), max(Neg_vals));

fprintf('\n=== Comparison of Three Evaluation Methods ===\n');
fprintf('Method\t\tOptimal α\tFWHM(nm)\tNegative Value\t\tPerformance Metric\n');
fprintf('----\t\t-----\t--------\t----\t\t------\n');
fprintf('1.Weighted Sum\t\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(1,1), optimal_results(1,2), optimal_results(1,3), optimal_results(1,4));
fprintf('2.Geometric Mean\t\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(2,1), optimal_results(2,2), optimal_results(2,3), optimal_results(2,4));
fprintf('3.Penalty Term\t\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(3,1), optimal_results(3,2), optimal_results(3,3), optimal_results(3,4));

% Analysis of method characteristics
fprintf('\n=== Method Characteristics Analysis ===\n');
fprintf('Method 1 (Weighted Sum): Flexible weight adjustment to balance FWHM and negative components according to application needs.\n');
fprintf('Method 2 (Geometric Mean): Equally sensitive to FWHM and negative components, requiring both to be small for good performance.\n');
fprintf('Method 3 (Penalty Term): Imposes hard constraints on negative components, suitable for applications sensitive to artifacts.\n');

elapsedTime = toc;
fprintf('\nCalculation completed！\n');
fprintf('The runtime is: %.4f s\n', elapsedTime);