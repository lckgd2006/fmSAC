%% Comparison between fmSAC and fmSAC+
% Investigate the influence of modulation contrast m_d on fmSAC
% Optimized version for calculating only center row data - Struct parameter version
clc; clear; close all;
addpath(genpath('PSF'));
addpath(genpath('CSV'));

%% Parameter initialization - Organized using structure
params = struct();

% Physical constants
params.h = 6.626e-34;        % Planck constant
params.c = 3e10;             % Speed of light (unit: cm)
params.lambda_s = 532e-7;    % Excitation wavelength (unit: cm)
params.lambda_d = 488e-7;    % Competing wavelength (unit: cm)

% Rate constants
params.k_isc = 1.1e6;
params.k_t = 0.49e6;
params.k0 = 2.56e8;
params.c1 = 1 + params.k_isc/params.k_t;

% Absorption cross-section
params.sigma_s = 2.7e-16;                      % Absorption cross-section of excitation light (for 532 nm wavelength)
params.sigma_d = params.sigma_s * 0.512063188; % Absorption cross-section of 488 nm wavelength

% Light intensity parameters
params.I_s = 10e3;           % Corresponding light intensity 10k W/cm²
params.I_d = 500e3;          % Corresponding light intensity 500 kW/cm²

% Frequency parameters
params.f1 = 10e3;            % f1 frequency
params.f2 = 15e3;            % f2 frequency

% Time parameters
params.interval = 10e-6;
params.t = 0:params.interval:1-params.interval;
params.n_time = length(params.t);

% Modulation contrast parameters
params.m_s = 0.1;            % Excitation modulation contrast
params.m_d_1 = 0.3;          % Competing light 1 modulation contrast
params.m_d_2 = 0.6;          % Competing light 2 modulation contrast
params.m_d_3 = 0.9;          % Competing light 3 modulation contrast
params.m_d_4 = 1.0;          % Competing light 4 modulation contrast

% Calculate relevant constants
params.const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
params.const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

% Frequency index
params.freq_res = (1/params.interval)/params.n_time;
params.f1_idx = round(params.f1/params.freq_res) + floor(params.n_time/2) + 1;
params.f2_idx = round(params.f2/params.freq_res) + floor(params.n_time/2) + 1;
params.f1_idx = min(max(params.f1_idx, 1), params.n_time);
params.f2_idx = min(max(params.f2_idx, 1), params.n_time);
params.center_freq_idx = floor(params.n_time/2) + 1;

%% Check GPU availability
if gpuDeviceCount > 0
    params.useGPU = true;
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
else
    params.useGPU = false;
    fprintf('Using CPU for calculation\n');
end

%% Load PSF data
try
    a = load('I_exc532_501.mat');
    I1 = a.result.PSF;
    b = load('I_hexc488_501.mat');
    I2 = b.result.PSF;
    
    [LL, MM] = size(I1);
    
    % Normalization
    I1 = I1 / max(I1(:));
    I2 = I2 / max(I2(:));
    
    % Scale light intensity
    I_exc = params.I_s * I1;
    I_hexc = params.I_d * I2;
    
catch ME
    error('Failed to load PSF files: %s', ME.message);
end

%% Extract only center row data
center_row = ceil(LL/2);                   % Row 251
fprintf('Calculating only center row data (Row %d)...\n', center_row);

% Extract center row data
I_exc_center = I_exc(center_row, :);
I_hexc_center = I_hexc(center_row, :);
num_points = length(I_exc_center);

%% Vectorized calculation of center row data
fprintf('Starting calculation of center row data (%d points)...\n', num_points);

% Transfer data to GPU if available
if params.useGPU
    I_exc_center_gpu = gpuArray(I_exc_center);
    I_hexc_center_gpu = gpuArray(I_hexc_center);
    t_gpu = gpuArray(params.t);
else
    I_exc_center_gpu = I_exc_center;
    I_hexc_center_gpu = I_hexc_center;
    t_gpu = params.t;
end

% Precompute cosine signals
cos_f1 = cos(2*pi*params.f1*t_gpu);
cos_f2 = cos(2*pi*params.f2*t_gpu);

% Calculate rate constant vectors
k_s_vector = params.const_s * I_exc_center_gpu;
k_d_vector = params.const_d * I_hexc_center_gpu;

% Traditional SAC calculation (vectorized)
y_SAC = k_s_vector ./ (params.c1*k_s_vector + params.k0 + params.c1*k_d_vector);

% Reshape to column vectors for vectorized calculation
k_s_col = reshape(k_s_vector, [], 1);
k_d_col = reshape(k_d_vector, [], 1);

% Expand dimensions for vectorized calculation
k_s_expanded = reshape(k_s_col, [num_points, 1]);
k_d_expanded = reshape(k_d_col, [num_points, 1]);

% Precompute expanded cosine signals
cos_f1_expanded = reshape(cos_f1, [1, params.n_time]);
cos_f2_expanded = reshape(cos_f2, [1, params.n_time]);

fprintf('Calculating modulation signals...\n');

% Vectorized calculation of y_s_1, y_s_2 and y_s_3
numerator = k_s_expanded .* (1 + params.m_s * cos_f1_expanded);
denominator_1= params.c1 * (k_s_expanded .* (1 + params.m_s * cos_f1_expanded) + ...
                   k_d_expanded .* (1 + params.m_d_1 * cos_f2_expanded)) + params.k0;
y_s_1 = numerator ./ denominator_1;

denominator_2= params.c1 * (k_s_expanded .* (1 + params.m_s * cos_f1_expanded) + ...
                   k_d_expanded .* (1 + params.m_d_2 * cos_f2_expanded)) + params.k0;
y_s_2 = numerator ./ denominator_2;

denominator_3 = params.c1 * (k_s_expanded .* (1 + params.m_s * cos_f1_expanded) + ...
                        k_d_expanded .* (1 + params.m_d_3 * cos_f2_expanded)) + params.k0;
y_s_3 = numerator ./ denominator_3;

denominator_4 = params.c1 * (k_s_expanded .* (1 + params.m_s * cos_f1_expanded) + ...
                        k_d_expanded .* (1 + params.m_d_3 * cos_f2_expanded)) + params.k0;
y_s_4 = numerator ./ denominator_4;

fprintf('Performing FFT analysis...\n');

% Batch FFT calculation
if params.useGPU
    f_fft_1 = fft(y_s_1, [], 2);
    f_fft_shift_1 = fftshift(f_fft_1, 2);
    result_1 = abs(f_fft_shift_1) ./ max(abs(f_fft_shift_1), [], 2);

    f_fft_2 = fft(y_s_2, [], 2);
    f_fft_shift_2 = fftshift(f_fft_2, 2);
    result_2 = abs(f_fft_shift_2) ./ max(abs(f_fft_shift_2), [], 2);
    
    f_fft_3 = fft(y_s_3, [], 2);
    f_fft_shift_3 = fftshift(f_fft_3, 2);
    result_3 = abs(f_fft_shift_3) ./ max(abs(f_fft_shift_3), [], 2);

    f_fft_4 = fft(y_s_4, [], 2);
    f_fft_shift_4 = fftshift(f_fft_4, 2);
    result_4 = abs(f_fft_shift_4) ./ max(abs(f_fft_shift_4), [], 2);
else
    % Vectorized FFT on CPU
    f_fft_1 = fft(y_s_11, [], 2);
    f_fft_shift_1 = fftshift(f_fft_1, 2);
    result_1 = abs(f_fft_shift_1) ./ max(abs(f_fft_shift_1), [], 2);

    f_fft_2 = fft(y_s_2, [], 2);
    f_fft_shift_2 = fftshift(f_fft_2, 2);
    result_2 = abs(f_fft_shift_2) ./ max(abs(f_fft_shift_2), [], 2);
    
    f_fft_3 = fft(y_s_3, [], 2);
    f_fft_shift_3 = fftshift(f_fft_3, 2);
    result_3 = abs(f_fft_shift_3) ./ max(abs(f_fft_shift_3), [], 2);

    f_fft_4 = fft(y_s_4, [], 2);
    f_fft_shift_4 = fftshift(f_fft_3, 2);
    result_4 = abs(f_fft_shift_4) ./ max(abs(f_fft_shift_4), [], 2);
end

% Calculate frequency component proportion (vectorized)
sumx_1 = (sum(result_1, 2) - result_1(:, params.center_freq_idx)) / 2;
sumx_2 = (sum(result_2, 2) - result_2(:, params.center_freq_idx)) / 2;
sumx_3 = (sum(result_3, 2) - result_3(:, params.center_freq_idx)) / 2;
sumx_4 = (sum(result_4, 2) - result_4(:, params.center_freq_idx)) / 2;

% Extract frequency components
sig_fund_1 = result_1(:, params.f1_idx) ./ sumx_1;
sig_fund_2 = result_2(:, params.f1_idx) ./ sumx_2;
sig_fund_3 = result_3(:, params.f1_idx) ./ sumx_3;
sig_fund_4 = result_4(:, params.f1_idx) ./ sumx_4;

sig_harm_1 = result_1(:, params.f2_idx) ./ sumx_1;
sig_harm_2 = result_2(:, params.f2_idx) ./ sumx_2;
sig_harm_3 = result_3(:, params.f2_idx) ./ sumx_3;
sig_harm_4 = result_4(:, params.f2_idx) ./ sumx_4;

%% Transfer data back to CPU if GPU was used
if params.useGPU
    fprintf('Transferring data from GPU back to CPU...\n');
    y_SAC = gather(y_SAC);
    sig_fund_1 = gather(sig_fund_1);
    sig_fund_2 = gather(sig_fund_2);
    sig_fund_3 = gather(sig_fund_3);
    sig_fund_4 = gather(sig_fund_4);

    sig_harm_1 = gather(sig_harm_1);
    sig_harm_2 = gather(sig_harm_2);
    sig_harm_3 = gather(sig_harm_3);
    sig_harm_4 = gather(sig_harm_4);
end

%% Calculate fmSAC_1, fmSAC_2 and fmSAC_3
fprintf('Calculating fmSAC_1, fmSAC_2 and fmSAC_3...\n');

% Use the differential coefficient at modulation coefficient m_d=1.0 for all modulation coefficients
alpha_matrix = sig_fund_4 ./ sig_harm_4;
alpha_val = min(alpha_matrix);

fmSAC_1 = sig_fund_1 - alpha_val * sig_harm_1;
fmSAC_2 = sig_fund_2 - alpha_val * sig_harm_2;
fmSAC_3 = sig_fund_3 - alpha_val * sig_harm_3;

% Align fmSAC_1/fmSAC_2/fmSAC_3 at the bottom
fmSAC_1 = fmSAC_1-min(fmSAC_1);
fmSAC_2 = fmSAC_2-min(fmSAC_2);
fmSAC_3 = fmSAC_3-min(fmSAC_3);

% Normalize traditional SAC
y_SAC = y_SAC / max(y_SAC);

%% Remove side lobes (only for center row)
% Create position vector
x_pos = 1:num_points;
center_pos = ceil(num_points/2);
radius = abs(x_pos - center_pos);

min_val_1 = min(fmSAC_1);
min_val_2 = min(fmSAC_2);
min_val_3 = min(fmSAC_3);
% Find positions of all minimum value points
threshold_1 = find(fmSAC_1 == min_val_1);
threshold_2 = find(fmSAC_2 == min_val_2);
threshold_3 = find(fmSAC_3 == min_val_3);

% Remove side lobes using relative threshold
% Coefficient 0.27 is selected based on 501×501 case, needs adjustment according to size
fmSAC_1(radius > abs(threshold_1 - center_pos)) = min(fmSAC_1);
fmSAC_2(radius > abs(threshold_2 - center_pos)) = min(fmSAC_2);
fmSAC_3(radius > abs(threshold_3 - center_pos)) = min(fmSAC_3);

% Calculate FWHM (Full Width at Half Maximum)
half_max = max(y_SAC) / 2;
half_index = find(y_SAC >= half_max);
FWHM = length(half_index); 

half_max_1 = max(fmSAC_1) / 2;
half_index_1 = find(fmSAC_1 >= half_max_1);
FWHM_1 = length(half_index_1); 

half_max_2 = max(fmSAC_2) / 2;
half_index_2 = find(fmSAC_2 >= half_max_2);
FWHM_2 = length(half_index_2); 

half_max_3 = max(fmSAC_3) / 2;
half_index_3 = find(fmSAC_3 >= half_max_3);
FWHM_3 = length(half_index_3); 

%% Plotting - Use vector data directly
fprintf('Generating plots...\n');

% Figure 1: fmSAC_1
figure(1);
set(gcf, 'Position', [100, 100, 1400, 500]);  % Figure window settings (pixels)
plot(sig_fund_1, 'linewidth', 2, 'DisplayName', 'Fund Freq_1(f_{1})');
hold on;
plot(sig_harm_1, 'linewidth', 2, 'DisplayName', 'Harm Freq_1(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_1, 'linewidth', 2, 'DisplayName', 'fmSAC_1');
hold off;
box on;
% Plot beautification
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_1 (CM_2 = %.1f)', params.m_d_1),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 2: fmSAC_2
figure(2);
set(gcf, 'Position', [100, 100, 1400, 500]);  % Figure window settings (pixels)
plot(sig_fund_2, 'linewidth', 2, 'DisplayName', 'Fund Freq_2(f_{1})');
hold on;
plot(sig_harm_3, 'linewidth', 2, 'DisplayName', 'Harm Freq_2(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_2, 'linewidth', 2, 'DisplayName', 'fmSAC_2');
hold off;
box on;
% Plot beautification
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_2 (CM_2 = %.1f)', params.m_d_2),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 3: fmSAC_3
figure(3);
set(gcf, 'Position', [100, 100, 1400, 500]);  % Figure window settings (pixels)
plot(sig_fund_3, 'linewidth', 2, 'DisplayName', 'Fund Freq_3(f_{1})');
hold on;
plot(sig_harm_3, 'linewidth', 2, 'DisplayName', 'Harm Freq_3(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_3, 'linewidth', 2, 'DisplayName', 'fmSAC_3');
hold off;
box on;
% Plot beautification
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_3 (CM_2 = %.1f)', params.m_d_3),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 4: Comparison
figure(4);
set(gcf, 'Position', [100, 100, 1400, 750]);  % Figure window settings (pixels)
plot(y_SAC,'Color',[0.85,0.33,0.1],'linewidth',2,'DisplayName','Conventional SAC');
hold on;
plot(fmSAC_1,'Color',[0.47,0.67,0.19],'linewidth',2,'DisplayName',sprintf('fmSAC_1 (CM_2=%.1f)',params.m_d_1));
plot(fmSAC_2,'Color',[0.12,0.47,0.71],'linewidth',2,'DisplayName',sprintf('fmSAC_2 (CM_2=%.1f)',params.m_d_2));
plot(fmSAC_3,'Color',[0.58,0.4,0.74],'linewidth',2,'DisplayName',sprintf('fmSAC_3 (CM_2=%.1f)',params.m_d_3));
% Calculate and plot FWHM horizontal lines
% Conventional SAC
if ~isempty(half_index)
    x_start_SAC = min(half_index);
    x_end_SAC = max(half_index);
    plot([x_start_SAC,x_end_SAC],[half_max,half_max],'--','Color',...
        [0.85,0.33,0.1],'LineWidth',1.5,'DisplayName',sprintf('FWHM=%.1f nm (Conventional SAC)',FWHM));
end

% fmSAC_1
if ~isempty(half_index_1)
    x_start_1 = min(half_index_1);
    x_end_1 = max(half_index_1);
    plot([x_start_1,x_end_1], [half_max_1,half_max_1],'--','Color',...
        [0.47,0.67,0.19],'LineWidth',1.5,'DisplayName',sprintf('FWHM_1=%.1f nm (fmSAC_1)',FWHM_1));
end

% fmSAC_2
if ~isempty(half_index_2)
    x_start_2 = min(half_index_2);
    x_end_2 = max(half_index_2);
    plot([x_start_2,x_end_2],[half_max_2,half_max_2],'--','Color',...
        [0.12,0.47,0.71],'LineWidth',1.5,'DisplayName',sprintf('FWHM_2=%.1f nm (fmSAC_2)',FWHM_2));
end

% fmSAC_3
if ~isempty(half_index_3)
    x_start_3 = min(half_index_3);
    x_end_3 = max(half_index_3);
    plot([x_start_3,x_end_3],[half_max_3,half_max_3],'--','Color',...
        [0.58,0.4,0.74],'LineWidth',1.5,'DisplayName',sprintf('FWHM_3=%.1f nm (fmSAC_3)',FWHM_3));
end

hold off;
box on;
% Plot beautification
xlim([0, 500]);
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Normalized Intensity (a.u.)', 'FontWeight', 'bold', 'FontSize', 24);
xlabel('Position (nm)','FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;


% Display parameter summary
fprintf('\n=== Parameter Summary ===\n');
fprintf('Physical Parameters:\n');
fprintf('  Excitation wavelength: %.1f nm, Competing wavelength: %.1f nm\n', params.lambda_s*1e7, params.lambda_d*1e7);
fprintf('  Excitation intensity: %.1f kW/cm², Competing intensity: %.1f kW/cm²\n', params.I_s/1e3, params.I_d/1e3);
fprintf('Modulation Parameters:\n');
fprintf('  Excitation modulation contrast m_s: %.1f\n', params.m_s);
fprintf('  Competing 1 modulation contrast m_d: %.1f (fmSAC_1)\n', params.m_d_1);
fprintf('  Competing 2 modulation contrast m_d: %.1f (fmSAC_2)\n', params.m_d_2);
fprintf('  Competing 3 modulation contrast m_d: %.1f (fmSAC_3)\n', params.m_d_3);
fprintf('  Competing 1 FWHM: %.1f nm (FWHM_1)\n', FWHM_1);
fprintf('  Competing 2 FWHM: %.1f nm (FWHM_2)\n', FWHM_2);
fprintf('  Competing 3 FWHM: %.1f nm (FWHM_3)\n', FWHM_3);
fprintf('  Conventional fmSAC FWHM: %.1f nm (FWHM)\n', FWHM);
fprintf('Frequency Parameters:\n');
fprintf('  f1: %.1f kHz, f2: %.1f kHz\n', params.f1/1e3, params.f2/1e3);
fprintf('Calculation Optimization:\n');
fprintf('  Number of calculation points reduced from %d to %d, speedup ratio about %.1fx\n', ...
        LL*MM, num_points, (LL*MM)/num_points);
fprintf('Analysis Completed!\n');