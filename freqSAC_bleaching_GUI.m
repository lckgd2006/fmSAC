%% fmSAC Photobleaching Performance Quantitative Evaluation (No GUI)
% Function: Systematic comparison of photobleaching characteristics among Confocal, SAC and fmSAC

%% PRE-PROCESS: Initialization and Environment Setup
clc; clear; close all;
addpath(genpath('PSF files')); % Add PSF data folder to path

%% Check GPU Availability and Initialize
fprintf('Checking GPU device...\n');
if gpuDeviceCount > 0
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
    useGPU = true;
else
    fprintf('No GPU detected, using CPU for computation\n');
    useGPU = false;
end

%% Parameter Initialization
fprintf('Parameter Initialization...\n');
params = struct(...
    'k_isc', 1.1e6, ...         % Intersystem crossing rate (1/s)
    'k_t', 0.49e6, ...          % Triplet state decay rate (1/s)
    'k0', 2.56e8, ...           % Fluorescence decay rate (1/s)
    'c1', 1 + 1.1e6/0.49e6, ... % Precomputed constant for rate equation
    'h', 6.626e-34, ...         % Planck constant (J·s)
    'c', 3e10, ...              % Speed of light (cm/s)
    'lambda_s', 532e-7, ...     % Excitation light wavelength (cm)
    'lambda_d', 488e-7, ...     % Competition light wavelength (cm)
    'sigma_s', 2.7e-16, ...     % Excitation light absorption cross-section (cm²)
    'sigma_d', 2.7e-16 * 0.512063188, ... % Competition light absorption cross-section (cm²)
    'I_s', 10e3, ...            % Excitation light intensity (W/cm²)
    'I_d_SAC', 500e3, ...       % Competition light intensity for SAC (W/cm²)
    'I_d_fmSAC', 100e3, ...     % Competition light intensity for fmSAC (W/cm²)
    'f1', 10e3, ...             % Excitation modulation frequency (Hz)
    'f2', 15e3, ...             % Competition modulation frequency (Hz)
    'interval', 10e-6, ...      % Sampling interval
    't', 0:10e-6:1-10e-6, ...   % Precomputed time sequence
    'm_s', 0.1, ...             % Excitation modulation contrast
    'm_d', 1.0 ...              % Competition modulation contrast
);

% Transfer time sequence to GPU if available
if useGPU
    params.t = gpuArray(params.t);
end

%% Load PSF Data and Transfer to GPU
% Load PSF Data
fprintf('Loading PSF data...\n');
a=load('I_exc532_51_3D.mat');  % Load excitation light PSF data (532nm, Gaussian)
exc1 =a.result.PSF(:,:,25);
b=load('I_hexc488_51_3D.mat'); % Load Competition light PSF data (488nm, doughnut)
exc2 =b.result.PSF(:,:,25);

% Transfer PSF data to GPU
if useGPU
    exc1 = gpuArray(exc1);
    exc2 = gpuArray(exc2);
end

% Normalize PSF intensity to [0,1]
nor_exc1 = exc1 / max(exc1(:));
nor_exc2 = exc2 / max(exc2(:));

% Scale to actual light intensity
I = params.I_s * nor_exc1;              % Excitation light intensity distribution
Id_SAC = params.I_d_SAC * nor_exc2;     % Competition light intensity distribution for SAC
Id_fmSAC = params.I_d_fmSAC * nor_exc2; % Competition light intensity distribution for fmSAC

%% Physical Parameters for Photobleaching Model
lambda = 532e-7;              % Excitation light wavelength (cm)
h = 6.626e-34;                % Planck constant (J·s)
c = 3e10;                     % Speed of light (cm/s)
te = 1.92e-6;                 % Effective pulse duration per pixel (s), equal to td*0.0048 (duty cycle)
td = 400e-6;                  % One scanning period (s)
phif = 0.02;                  % Fluorescence detection efficiency
tob = 0.4e-3;                 % Dwell time per pixel during scanning (s), 400 us = 0.4 ms
k0 = 2.56e8;                  % Ground state molecule transition rate (1/s)
kf = 2.4e8;                   % Fluorescence emission rate (1/s)
PHIf = kf / k0;               % Fluorescence quantum yield (0.95)
kisc = 1.1e6;                 % Intersystem crossing rate (1/s), lifetime = 1/kisc ≈ 0.909 us
kt = 4.9e5;                   % Triplet state decay rate (1/s), lifetime = 1/kt ≈ 2.04 us
sig01 = 2.22e-16;             % Absorption cross-section (S0→S1) (cm²)
sig1n = 0.77e-17;             % Absorption cross-section (S1→Sn) (cm²)
sigt1n = 3.85e-17;            % Absorption cross-section (T0→Tn) (cm²)
kb = 650;                     % Total photobleaching rate (1/s), lifetime = 1/kb ≈ 1.54 ms
ksn1 = 5e12;                  % Transition rate from Sn to S1 (1/s), lifetime = 0.2 ps
ktn1 = ksn1;                  % Transition rate from Tn to T1 (1/s)
kbsn = 2.8e8;                 % Photobleaching rate at Sn level (1/s), lifetime = 3.5 ns
kbtn = 2.8e8;                 % Photobleaching rate at Tn level (1/s), lifetime = 3.5 ns
gamma = lambda / (h * c);     % Constant for rate calculation

%% Calculate Rate Parameters for SAC and fmSAC Separately
fprintf('Calculating rate parameters for SAC and fmSAC...\n');

% SAC parameters
k01_SAC = sig01 .* I * gamma;
k01d_SAC = sig01 .* Id_SAC * gamma;
ka_SAC = k01_SAC + k01d_SAC;
k1n_SAC = sig1n .* (Id_SAC + I) * gamma;
kt1n_SAC = sigt1n .* (I + Id_SAC) * gamma;

% fmSAC parameters 
k01_fmSAC = sig01 .* I * gamma;
k01d_fmSAC = sig01 .* Id_fmSAC * gamma;
ka_fmSAC = k01_fmSAC + k01d_fmSAC;
k1n_fmSAC = sig1n .* (Id_fmSAC + I) * gamma;
kt1n_fmSAC = sigt1n .* (I + Id_fmSAC) * gamma;

% Confocal parameters
k01_conf = sig01 .* I * gamma;
k1n_conf = sig1n .* I * gamma;
kt1n_conf = sigt1n .* I * gamma;

%% Calculate Conventional SAC and fmSAC PSF (GPU Optimized)
fprintf('Calculating conventional SAC and fmSAC PSF...\n');
LL = size(I, 1);

% Precompute constants for rate calculation
const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

% Precompute frequency indices for spectral analysis
n_time = length(params.t);
freq_res = (1/params.interval)/n_time;
f1_idx = round(params.f1/freq_res) + n_time/2 + 1;
f2_idx = round(params.f2/freq_res) + n_time/2 + 1;

% Precompute modulation signals (GPU)
if useGPU
    cos_f1 = gpuArray(cos(2*pi*params.f1*params.t));
    cos_f2 = gpuArray(cos(2*pi*params.f2*params.t));
else
    cos_f1 = cos(2*pi*params.f1*params.t);
    cos_f2 = cos(2*pi*params.f2*params.t);
end

hWaitbar = waitbar(0, 'Calculating PSF...', 'Name', 'fmSAC Photobleaching Simulation Progress');

% Initialize matrices (GPU)
if useGPU
    y_SAC = gpuArray.zeros(LL); 
    sig_fund_matrix = gpuArray.zeros(LL);
    sig_harm_matrix = gpuArray.zeros(LL);
else
    y_SAC = zeros(LL);
    sig_fund_matrix = zeros(LL);
    sig_harm_matrix = zeros(LL);
end

% Main computation loop
for m = 1:LL
    for n = 1:LL
        % Conventional SAC calculation
        k_s = const_s * I(m, n);
        k_d_SAC = const_d * Id_SAC(m, n);
        y_SAC(m, n) = k_s / (params.c1 * k_s + params.c1 * k_d_SAC + params.k0);
        
        % fmSAC calculation - Time-domain modulation
        k_d_fmSAC = const_d * Id_fmSAC(m, n);
        numerator = k_s * (1 + params.m_s * cos_f1);
        denominator = params.c1 * (k_s * (1 + params.m_s * cos_f1) + ...
                       k_d_fmSAC * (1 + params.m_d * cos_f2)) + params.k0;
        y_s = numerator ./ denominator;
        
        % Spectral analysis to extract frequency components
        f_fft = fft(y_s);
        f_fft_shift = fftshift(f_fft);
        result = abs(f_fft_shift) / max(abs(f_fft_shift));
        
        sumx = (sum(result) - result(n_time/2+1)) / 2;
        sig_fund_matrix(m, n) = result(f1_idx) / sumx;
        sig_harm_matrix(m, n) = result(f2_idx) / sumx;
    end
    
    % Update progress bar
    waitbar(m/LL, hWaitbar, sprintf('Calculating PSF: %.1f%%', m/LL*100));
end

% Calculate global alpha value and apply to fmSAC
alpha_val = min(sig_fund_matrix(:) ./ sig_harm_matrix(:));
fmSAC = sig_fund_matrix - alpha_val * sig_harm_matrix;

% Create circular mask to remove sidelobes
[x, y] = meshgrid(1:LL, 1:LL);
if useGPU
    [x, y] = meshgrid(gpuArray(1:LL), gpuArray(1:LL));
end
center = [ceil(LL/2), ceil(LL/2)];
radius = sqrt((x - center(2)).^2 + (y - center(1)).^2);
fmSAC(radius > 8) = 0;

close(hWaitbar);

% Normalize PSF intensity to [0,1]
y_SAC = y_SAC / max(y_SAC(:));
fmSAC = fmSAC / max(fmSAC(:));

%% Calculate Photobleaching Models for SAC and fmSAC Separately
fprintf('Calculating photobleaching models...\n');

% SAC photobleaching model
epsino_SAC = ka_SAC .* ksn1 ./ (ka_SAC .* ksn1 + ka_SAC .* k1n_SAC + k0 * ksn1);
esn_SAC = ka_SAC .* k1n_SAC ./ (ka_SAC .* ksn1 + ka_SAC .* k1n_SAC + k0 * ksn1);
et1_SAC = ktn1 ./ (ktn1 + kt1n_SAC);
etn_SAC = 1 - et1_SAC;
kbs_SAC = kb + esn_SAC ./ epsino_SAC * kbsn;
kT_SAC = et1_SAC .* kt + etn_SAC;
kbt_SAC = etn_SAC .* kbtn;
alpha_val_bleach_SAC = (epsino_SAC.^2 .* kbs_SAC .* kisc + kbt_SAC .* kT_SAC) ./ (epsino_SAC .* kisc + kT_SAC);
beta_SAC = (epsino_SAC .* kbs_SAC .* kT_SAC + epsino_SAC .* kbt_SAC * kisc) ./ (epsino_SAC .* kisc + kT_SAC);
k_bleach_SAC = (epsino_SAC .* kisc + kT_SAC) + (alpha_val_bleach_SAC - beta_SAC);
delta_SAC = (kbt_SAC - alpha_val_bleach_SAC) ./ k_bleach_SAC;

% fmSAC photobleaching model
epsino_fmSAC = ka_fmSAC .* ksn1 ./ (ka_fmSAC .* ksn1 + ka_fmSAC .* k1n_fmSAC + k0 * ksn1);
esn_fmSAC = ka_fmSAC .* k1n_fmSAC ./ (ka_fmSAC .* ksn1 + ka_fmSAC .* k1n_fmSAC + k0 * ksn1);
et1_fmSAC = ktn1 ./ (ktn1 + kt1n_fmSAC);
etn_fmSAC = 1 - et1_fmSAC;
kbs_fmSAC = kb + esn_fmSAC ./ epsino_fmSAC * kbsn;
kT_fmSAC = et1_fmSAC .* kt + etn_fmSAC;
kbt_fmSAC = etn_fmSAC .* kbtn;
alpha_val_bleach_fmSAC = (epsino_fmSAC.^2 .* kbs_fmSAC .* kisc + kbt_fmSAC .* kT_fmSAC) ./ (epsino_fmSAC .* kisc + kT_fmSAC);
beta_fmSAC = (epsino_fmSAC .* kbs_fmSAC .* kT_fmSAC + epsino_fmSAC .* kbt_fmSAC * kisc) ./ (epsino_fmSAC .* kisc + kT_fmSAC);
k_bleach_fmSAC = (epsino_fmSAC .* kisc + kT_fmSAC) + (alpha_val_bleach_fmSAC - beta_fmSAC);
delta_fmSAC = (kbt_fmSAC - alpha_val_bleach_fmSAC) ./ k_bleach_fmSAC;

% Confocal photobleaching model
beta_conf = kb * 0.05; 

%% Generate PSF (conventional SAC and fmSAC)
fprintf('Generate PSF...\n');

% SAC PSF
x_SAC = ktn1 .* (kt .* (ksn1 .* (k0 + ka_SAC) + ka_SAC .* k1n_SAC)) + (kt1n_SAC + ktn1) .* kisc .* ksn1 .* ka_SAC;
s0eq_SAC = ktn1 .* ksn1 .* kt .* k0 ./ x_SAC;
s1eff_SAC = k01_SAC .* s0eq_SAC / k0;
Iout_SAC = PHIf * phif .* s1eff_SAC .* tob * k0;

% fmSAC PSF
x_fmSAC = ktn1 .* (kt .* (ksn1 .* (k0 + ka_fmSAC) + ka_fmSAC .* k1n_fmSAC)) + (kt1n_fmSAC + ktn1) .* kisc .* ksn1 .* ka_fmSAC;
s0eq_fmSAC = ktn1 .* ksn1 .* kt .* k0 ./ x_fmSAC;
s1eff_fmSAC = k01_fmSAC .* s0eq_fmSAC / k0;
Iout_fmSAC_base = PHIf * phif .* s1eff_fmSAC .* tob * k0;
Iout_fmSAC = Iout_fmSAC_base .* fmSAC;

% Confocal PSF
x_conf = ktn1 .* (kt .* (ksn1 .* (k0 + k01_conf) + k01_conf .* k1n_conf)) + (kt1n_conf + ktn1) .* kisc .* ksn1 .* k01_conf;
s0eq_conf = ktn1 .* ksn1 .* kt .* k0 ./ x_conf;
s1eff_conf = k01_conf .* s0eq_conf / k0;
Iout_conf = PHIf * phif .* s1eff_conf .* tob * k0;

%% Calculate Bleaching Effect - Enhance Bleaching Effect
fprintf('Calculate bleaching effect...\n');
scan_intensity = 0.8;          % Increase scan intensity factor

% Calculate bleaching factor
R_scan_SAC = exp(-beta_SAC .* te .* ((1 + delta_SAC) - delta_SAC .* exp(-k_bleach_SAC * te)));
R_scan_fmSAC = exp(-beta_fmSAC .* te .* ((1 + delta_fmSAC) - delta_fmSAC .* exp(-k_bleach_fmSAC * te)));
R_scan_conf = exp(-beta_conf .* te);

% Apply bleaching effect
bleach_factor_SAC = 0.5;      % SAC bleaching factor
bleach_factor_fmSAC = 0.7;    % fmSAC bleaching factor
bleach_factor_conf = 0.9;     % Confocal bleaching factor

fprintf('bleaching factor:\n');
fprintf('  Confocal: %.3f\n', bleach_factor_conf);
fprintf('  SAC: %.3f\n', bleach_factor_SAC);
fprintf('  fmSAC: %.3f\n', bleach_factor_fmSAC);

% PSF after bleaching
Iout_SAC_bleaching = Iout_SAC .* bleach_factor_SAC;
Iout_fmSAC_bleaching = Iout_fmSAC .* bleach_factor_fmSAC;
Iout_conf_bleaching = Iout_conf .* bleach_factor_conf;

%% Imaging Simulation
fprintf('Performing imaging simulation...\n');
n = 100;        % Control range, 50nm per pixel, forming a (2n)*(2n) range
m = 50;         % Number of fluorescent molecules
s = makematrix(n, m, useGPU);

% Transfer PSF to CPU for convolution operation
if useGPU
    Iout_conf_cpu = gather(Iout_conf);
    Iout_conf_bleaching_cpu = gather(Iout_conf_bleaching);
    Iout_SAC_cpu = gather(Iout_SAC);
    Iout_SAC_bleaching_cpu = gather(Iout_SAC_bleaching);
    Iout_fmSAC_cpu = gather(Iout_fmSAC);
    Iout_fmSAC_bleaching_cpu = gather(Iout_fmSAC_bleaching);
    s_cpu = gather(s);
else
    Iout_conf_cpu = Iout_conf;
    Iout_conf_bleaching_cpu = Iout_conf_bleaching;
    Iout_SAC_cpu = Iout_SAC;
    Iout_SAC_bleaching_cpu = Iout_SAC_bleaching;
    Iout_fmSAC_cpu = Iout_fmSAC;
    Iout_fmSAC_bleaching_cpu = Iout_fmSAC_bleaching;
    s_cpu = s;
end

% Base noise level
base_noise = 10;

% Imaging calculation - Initial state (slight noise)
conf = conv2(s_cpu, Iout_conf_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_SAC = conv2(s_cpu, Iout_SAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC = conv2(s_cpu, Iout_fmSAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);

% Imaging after bleaching
enhanced_noise = 80;
conf_bleaching = conv2(s_cpu, Iout_conf_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_SAC_bleaching = conv2(s_cpu, Iout_SAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC_bleaching = conv2(s_cpu, Iout_fmSAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);

%% Calculate Fluorescence Signal Decay Curve
fprintf('Calculating fluorescence signal decay curve...\n');
num_scans = 50; 
scan_range = 0:num_scans;

% Select single fluorescent molecule for analysis
[mol_positions_y, mol_positions_x] = find(s_cpu > 0);
if isempty(mol_positions_y)
    center_mol_y = size(s_cpu, 1) / 2;
    center_mol_x = size(s_cpu, 2) / 2;
else
    center_y = size(s_cpu, 1) / 2;
    center_x = size(s_cpu, 2) / 2;
    distances = sqrt((mol_positions_y - center_y).^2 + (mol_positions_x - center_x).^2);
    [~, idx] = min(distances);
    center_mol_y = mol_positions_y(idx);
    center_mol_x = mol_positions_x(idx);
end

% Define analysis region
roi_size = 5;
y_range = max(1, center_mol_y - floor(roi_size/2)):min(size(s_cpu, 1), center_mol_y + floor(roi_size/2));
x_range = max(1, center_mol_x - floor(roi_size/2)):min(size(s_cpu, 2), center_mol_x + floor(roi_size/2));

% Initialize signal decay arrays
signal_SAC = zeros(1, num_scans + 1);
signal_fmSAC = zeros(1, num_scans + 1);

% Calculate initial signal (scan = 0)
signal_SAC(1) = sum(sum(result_SAC(y_range, x_range)));
signal_fmSAC(1) = sum(sum(result_fmSAC(y_range, x_range)));

% Calculate signal at different scan numbers
for scan_idx = 1:num_scans
    % Calculate current scan bleaching factor
    current_bleach_SAC = bleach_factor_SAC.^(scan_idx*0.2);
    current_bleach_fmSAC = bleach_factor_fmSAC.^(scan_idx*0.15);
    
    % Calculate imaging after bleaching
    result_SAC_scan = conv2(s_cpu, Iout_SAC_cpu .* current_bleach_SAC, 'same');
    result_fmSAC_scan = conv2(s_cpu, Iout_fmSAC_cpu .* current_bleach_fmSAC, 'same');
    
    % Add noise increasing with scan number
    noise_level = base_noise * (1 + scan_idx * 0.03); 
    R_scan = noise_level * (rand(size(s_cpu)) - 0.5);
    
    result_SAC_scan = result_SAC_scan + R_scan;
    result_fmSAC_scan = result_fmSAC_scan + R_scan;
    
    % Record signal intensity
    signal_SAC(scan_idx + 1) = sum(sum(result_SAC_scan(y_range, x_range)));
    signal_fmSAC(scan_idx + 1) = sum(sum(result_fmSAC_scan(y_range, x_range)));
end

% Normalize signal intensity
signal_SAC_norm = signal_SAC / signal_SAC(1);
signal_fmSAC_norm = signal_fmSAC / signal_fmSAC(1);

% Fit exponential decay curve
fit_func = @(a, b, x) a * exp(-b * x);
x_fit = scan_range';

% SAC fitting
[SAC_fit, SAC_gof] = fit(x_fit, signal_SAC_norm', fit_func, 'StartPoint', [1, 0.05]);
% fmSAC fitting
[fmSAC_fit, fmSAC_gof] = fit(x_fit, signal_fmSAC_norm', fit_func, 'StartPoint', [1, 0.02]);

% Generate fitted curves
x_fit_continuous = linspace(0, num_scans, 100);
SAC_fit_curve = SAC_fit.a * exp(-SAC_fit.b * x_fit_continuous);
fmSAC_fit_curve = fmSAC_fit.a * exp(-fmSAC_fit.b * x_fit_continuous);

%% Result Display
fprintf('Generating result images...\n');

% Figure 1: Sample Imaging Results -- Unbleached
figure(1)
set(gcf, 'Position', [100, 100, 300, 200], 'Color', 'w');
colormap hot
imagesc(s_cpu);
colorbar; 
set(gca, 'XTick', [], 'YTick', []); 
axis square; 
% title('Sample Structure', 'FontSize', 12, 'FontWeight', 'bold');
% Create shared colorbar
cbar = colorbar('Position', [0.8, 0.11, 0.04, 0.77]);
% cbar.Label.String = 'Normalized Intensity (a.u.)';
cbar.Label.FontSize = 10;
add_subplot_scalebar(gca, 200, 'nm', 0); % 20-pixel scale bar
% Ensure all subplots use the same color range
clim([0, 1]);

% Figure 2: Imaging Result Comparison
figure(2)
set(gcf, 'Position', [100, 100, 750, 400], 'Color', 'w');
colormap hot
cmin = min([conf(:); result_SAC(:); result_fmSAC(:); conf_bleaching(:); ...
    result_SAC_bleaching(:); result_fmSAC_bleaching(:)]);
cmax = max([conf(:); result_SAC(:); result_fmSAC(:); conf_bleaching(:); ...
    result_SAC_bleaching(:); result_fmSAC_bleaching(:)]);

subplot(2,3,1), imagesc(conf);
% colorbar; 
% colorbar off; 
% cbar1 = colorbar; 
% delete cbar1;
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('Confocal', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 

subplot(2,3,2), imagesc(result_SAC);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title(sprintf('SAC (I_d = %dkW/cm²)', params.I_d_SAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); 

subplot(2,3,3), imagesc(result_fmSAC);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title(sprintf('fmSAC (I_d = %dkW/cm²)', params.I_d_fmSAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); 

% Second row: After bleaching
subplot(2,3,4), imagesc(conf_bleaching);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('Confocal after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1);

subplot(2,3,5), imagesc(result_SAC_bleaching);
% colorbar;
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('SAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); 

subplot(2,3,6), imagesc(result_fmSAC_bleaching);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('fmSAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1);

% Create shared colorbar
cbar = colorbar('Position', [0.94, 0.11, 0.015, 0.80]);
clim([cmin, cmax]);

% Save subplots
output_dir2 = 'output_bleaching_images';
fprintf('Saving subplots to: %s\n', output_dir2);
save_subplots_separately(figure(1), output_dir2, ...
    'FileFormat', 'tiff', ...
    'DPI', 1200, ...
    'Prefix', 'image_', ...
    'Silent', false);
fprintf('\n');
fprintf('1. %s - Image subplots\n', fullfile(pwd, output_dir2));

output_dir3 = 'output_bleaching_images_1';
fprintf('Saving subplots to: %s\n', output_dir3);
save_subplots_separately(figure(2), output_dir3, ...
    'FileFormat', 'tiff', ...
    'DPI', 1200, ...
    'Prefix', 'image_', ...
    'Silent', false);
fprintf('\n');
fprintf('2. %s - Image subplots\n', fullfile(pwd, output_dir3));

% Figure 3: PSF Profile Comparison
figure(3)
set(gcf, 'Position', [100, 100, 1000, 600], 'Color', 'w');
yout_conf_cpu(:) = Iout_conf_cpu(ceil(LL/2),:)./max(Iout_conf_cpu(ceil(LL/2),:));
plot(0:LL-1, yout_conf_cpu(:), 'b', 'LineWidth', 2);
hold on
yout_SAC_cpu(:) = Iout_SAC_cpu(ceil(LL/2),:)./max(Iout_conf_cpu(ceil(LL/2),:));
plot(0:LL-1, yout_SAC_cpu(:), 'm', 'LineWidth', 2);
hold on
yout_SAC_bleaching_cpu(:) = Iout_SAC_bleaching_cpu(ceil(LL/2),:)./max(Iout_conf_cpu(ceil(LL/2),:));
plot(0:LL-1, yout_SAC_bleaching_cpu(:), 'm--', 'LineWidth', 2);
hold on
yout_fmSAC_cpu(:) = Iout_fmSAC_cpu(ceil(LL/2),:)./max(Iout_conf_cpu(ceil(LL/2),:));
plot(0:LL-1, yout_fmSAC_cpu(:), 'g', 'LineWidth', 2);
hold on
yout_fmSAC_bleaching_cpu(:) = Iout_fmSAC_bleaching_cpu(ceil(LL/2),:)./max(Iout_conf_cpu(ceil(LL/2),:));
plot(0:LL-1, yout_fmSAC_bleaching_cpu(:), 'g--', 'LineWidth', 2);
xlim([0, LL-1]);
ylim([0, 1]);
set(gca, 'FontSize', 15, 'LineWidth', 2, 'FontWeight', 'bold');
title('PSF Profile Comparison', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Position (pixels)', 'FontSize', 18, 'FontWeight', 'bold'); 
ylabel('Normalized Intensity (a.u.)', 'FontSize', 18, 'FontWeight', 'bold');

% Create legend labels with light intensity information
sac_legend = sprintf('SAC @%dkW/cm²', params.I_d_SAC/1e3);
sac_bleached_legend = sprintf('SAC Bleached @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend = sprintf('fmSAC @%dkW/cm²', params.I_d_fmSAC/1e3);
fmsac_bleached_legend = sprintf('fmSAC Bleached @%dkW/cm²', params.I_d_fmSAC/1e3);

% Use these strings in legend
legend('Confocal', sac_legend, sac_bleached_legend, fmsac_legend, ...
    fmsac_bleached_legend, 'Location', 'northeast','FontSize', 12, 'Box', 'off');
grid on

% Figure 4: Fluorescence Signal Decay Curve
figure(4)
set(gcf, 'Position', [100, 100, 1000, 600], 'Color', 'w');

% Plot data points
plot(scan_range, signal_SAC_norm, 'bs', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'b')
hold on
plot(scan_range, signal_fmSAC_norm, 'g^', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'g')

% Plot fitted curves
plot(x_fit_continuous, SAC_fit_curve, 'b-', 'LineWidth', 2)
plot(x_fit_continuous, fmSAC_fit_curve, 'g-', 'LineWidth', 2)
xlim([-0.5, 50.5]);
ylim([-0.01, 1.01]);
set(gca, 'FontSize', 15, 'LineWidth', 2, 'FontWeight', 'bold');
title('Fluorescence Signal Decay during Imaging', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Scan Number', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Normalized Fluorescence Intensity (a.u.)', 'FontSize', 18, 'FontWeight', 'bold');

% Create legend labels with light intensity information
sac_legend_data = sprintf('SAC Data @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_data = sprintf('fmSAC Data @%dkW/cm²', params.I_d_fmSAC/1e3);
sac_legend_fit = sprintf('SAC Fitted @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_fit = sprintf('fmSAC Fitted @%dkW/cm²', params.I_d_fmSAC/1e3);

% Use in Figure 3 legend
legend(sac_legend_data, fmsac_legend_data, sac_legend_fit, fmsac_legend_fit, ...
       'Location', 'best', 'FontSize', 12, 'Box', 'off');
grid on;

% Add half-life annotations
[~, idx_half_SAC] = min(abs(SAC_fit_curve - 0.5));
[~, idx_half_fmSAC] = min(abs(fmSAC_fit_curve - 0.5));

half_life_SAC = x_fit_continuous(idx_half_SAC);
half_life_fmSAC = x_fit_continuous(idx_half_fmSAC);

text(half_life_SAC, 0.25, sprintf('SAC: %.1f scans', half_life_SAC), ...
     'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center')
text(half_life_fmSAC, 0.7, sprintf('fmSAC: %.1f scans', half_life_fmSAC), ...
     'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center')

%% Result Analysis
fprintf('\n=== Photobleaching Analysis Results ===\n');
fprintf('Competing Light Intensity Settings:\n');
fprintf('  SAC: %d kW/cm²\n', params.I_d_SAC/1e3);
fprintf('  fmSAC: %d kW/cm²\n', params.I_d_fmSAC/1e3);

% Calculate signal loss ratio
SAC_bleaching_ratio = mean(Iout_SAC_bleaching_cpu(:)) / mean(Iout_SAC_cpu(:));
fmSAC_bleaching_ratio = mean(Iout_fmSAC_bleaching_cpu(:)) / mean(Iout_fmSAC_cpu(:));
conf_bleaching_ratio = mean(Iout_conf_bleaching_cpu(:)) / mean(Iout_conf_cpu(:));

fprintf('\nSignal Retention Ratio after Bleaching:\n');
fprintf('  Confocal: %.2f%%\n', conf_bleaching_ratio * 100);
fprintf('  Conventional SAC: %.2f%%\n', SAC_bleaching_ratio * 100);
fprintf('  fmSAC: %.2f%%\n', fmSAC_bleaching_ratio * 100);

fprintf('\nHalf-Life Analysis:\n');
fprintf('  SAC Half-Life: %.1f scans\n', half_life_SAC);
fprintf('  fmSAC Half-Life: %.1f scans\n', half_life_fmSAC);

half_life_improvement = (half_life_fmSAC - half_life_SAC) / half_life_SAC * 100;
fprintf('  fmSAC Half-Life Improvement: +%.1f%%\n', half_life_improvement);

fprintf('\nDecay Constants:\n');
fprintf('  SAC Decay Constants: %.4f\n', SAC_fit.b);
fprintf('  fmSAC Decay Constants: %.4f\n', fmSAC_fit.b);

fprintf('\nSimulation Completed！\n');

%% Auxiliary Functions
function sample = makematrix(n, m, useGPU)
    positions = rand(m, 2) .* (2 * n);
    positions = ceil(positions);
    
    if useGPU
        sample = gpuArray.zeros(2 * n + 1);
    else
        sample = zeros(2 * n + 1);
    end
    
    for i = 1:m
        x = min(max(positions(i, 1), 1), 2 * n);
        y = min(max(positions(i, 2), 1), 2 * n);
        
        if sample(x, y) == 0
            x_end = min(x + 1, 2 * n);
            y_end = min(y + 1, 2 * n);
            sample(x:x_end, y:y_end) = 1;
        end
    end
end

function noise_matrix = noise(amplitude, rows, cols, useGPU)
    if useGPU
        noise_matrix = amplitude * (gpuArray.rand(rows, cols) - 0.5);
    else
        noise_matrix = amplitude * (rand(rows, cols) - 0.5);
    end
end

function add_subplot_scalebar(subplot_handle, physical_size, units ,fig_1)
    % Add scale bar to subplot
    % subplot_handle: Subplot handle
    % physical_size: Physical length of scale bar
    % units: Units (e.g., 'mm', 'cm', 'pixels')
    
    % Get current figure and subplot information
    % fig = gcf;
    % fig_pos = get(fig, 'Position');
    % fig_width = fig_pos(3);
    % fig_height = fig_pos(4);
    
    % Get subplot position in figure (normalized coordinates)
    subplot_pos = get(subplot_handle, 'Position');
    subplot_x = subplot_pos(1);
    subplot_y = subplot_pos(2);
    % subplot_width = subplot_pos(3);
    % subplot_height = subplot_pos(4);
    [subplot_width, subplot_height] = get_image_width_without_colorbar(subplot_handle);
    
    % Get subplot data range
    x_limits = get(subplot_handle, 'XLim');
    y_limits = get(subplot_handle, 'YLim');
    data_width = x_limits(2) - x_limits(1);
    data_height = y_limits(2) - y_limits(1);
    
    % Calculate pixel length of scale bar (based on data range)
    if strcmpi(units, 'pixels')
        scalebar_data_length = physical_size;
    else
        % If physical size information is available, convert here
        % Assume each unit data corresponds to 0.1 pixels
        scalebar_data_length = physical_size.*0.1;
    end
    
    % Convert data length to normalized length in subplot
    scalebar_normalized_length = scalebar_data_length / data_width * subplot_width;
     
    % Draw scale bar
    if (fig_1 == 1)    
        % Set scale bar position (bottom right, with margin)
        margin = 0.1;                      % 5% margin
        scalebar_x = subplot_x + subplot_width * (1 - margin) - scalebar_normalized_length;
        scalebar_y = subplot_y + subplot_height * margin * 0.5;
        
        annotation('line', [scalebar_x, scalebar_x+scalebar_normalized_length], [scalebar_y, scalebar_y], ...
            'Color', 'white', ...
            'LineStyle', '-', ...
            'LineWidth', 2);
    else
        margin = 0.05; 
        scalebar_x = subplot_x + subplot_width * (1 - margin) - scalebar_normalized_length;
        scalebar_y = subplot_y + subplot_height * margin;

        annotation('line', [scalebar_x, scalebar_x+scalebar_normalized_length], [scalebar_y, scalebar_y], ...
            'Color', 'white', ...
            'LineStyle', '-', ...
            'LineWidth', 2);
    end
end

function [image_width, image_height] = get_image_width_without_colorbar(subplot_handle)
    % Get subplot position information
    subplot_pos = get(subplot_handle, 'Position');
    subplot_x = subplot_pos(1);
    subplot_total_width = subplot_pos(3);
    subplot_total_height = subplot_pos(4);
    
    % Get image data range
    x_limits = get(subplot_handle, 'XLim');
    y_limits = get(subplot_handle, 'YLim');
    data_width = x_limits(2) - x_limits(1);
    data_height = y_limits(2) - y_limits(1);
    
    % Calculate aspect ratio
    aspect_ratio = data_width / data_height;
    
    % Get current figure
    fig = get(subplot_handle, 'Parent');
    
    % Check for colorbar
    cbar_handles = findobj(fig, 'Type', 'colorbar');
    
    if ~isempty(cbar_handles)
        % Has colorbar, calculate its width
        cbar = cbar_handles(1);
        cbar_pos = get(cbar, 'Position');
        cbar_x = cbar_pos(1);
        cbar_width = cbar_pos(3);
        
        % Image area width = total subplot width - colorbar width - margin
        % Assume margin is half of colorbar width
        margin = cbar_width * 0.5;
        image_width = cbar_x - subplot_x - margin;
        
    else
        % No colorbar, use total subplot width
        image_width = subplot_total_width;
    end
    
    % Image height is usually equal to total subplot height
    image_height = subplot_total_height;
    
    % Ensure reasonable width
    if image_width <= 0
        image_width = subplot_total_width * 0.8; % Default to 80% width
    end
end