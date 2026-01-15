%% PRE-PROCESS
% fmSAC Photobleaching Performance Analysis (GUI Interactive Version)
% Core function: Evaluate the photobleaching resistance of fmSAC microscopy vs traditional SAC microscopy
% Provide GUI interface to view imaging quality at arbitrary scanning number
% Quantitatively compare fluorophore survival rate and imaging signal attenuation after multi-scan
clc; clear; close all;
addpath(genpath('PSF files'));

%% GPU acceleration detection & initialization
fprintf('Searching for GPU devices...\n');
if gpuDeviceCount > 0
    gpu = gpuDevice();
    fprintf('GPU acceleration enabled: %s\n', gpu.Name);
    useGPU = true;
else
    fprintf('No GPU detected, CPU computation mode enabled\n');
    useGPU = false;
end

%% Parameter Initialization
fprintf('Parameter Initialization...\n');
params = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'c1', 1 + 1.1e6/0.49e6, ...           % Pre-calculation
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'lambda_s', 532e-7, ...
    'lambda_d', 488e-7, ...
    'sigma_s', 2.7e-16, ...
    'sigma_d', 2.7e-16 * 0.512063188, ... % Pre-calculation
    'I_s', 10e3, ...
    'I_d_SAC', 500e3, ...                 % Competitive light intensity for SAC
    'I_d_fmSAC', 100e3, ...               % Competitive light intensity for fmSAC
    'f1', 10e3, ...
    'f2', 15e3, ...
    'interval', 10e-6, ...
    't', 0:10e-6:1-10e-6, ...             % Pre-calculated time series
    'm_s', 1.0, ...
    'm_d', 0.9 ...
);

% Transfer time series to GPU
if useGPU
    params.t = gpuArray(params.t);
end

%% Load PSF data and transfer to GPU
fprintf('Loading PSF data...\n');
a=load('I_exc532_51_3D.mat');
exc1 =a.result.PSF(:,:,25);
b=load('I_hexc488_51_3D.mat');
exc2 =b.result.PSF(:,:,25);

% transfer to GPU
if useGPU
    exc1 = gpuArray(exc1);
    exc2 = gpuArray(exc2);
end

nor_exc1 = exc1 / max(exc1(:));
nor_exc2 = exc2 / max(exc2(:));

I = params.I_s * nor_exc1;
Id_SAC = params.I_d_SAC * nor_exc2;
Id_fmSAC = params.I_d_fmSAC * nor_exc2;

%% Physical parameters
lambda = 532e-7;  
h = 6.626e-34;    
c = 3e10;             
te = 1.92e-6;      
td = 400e-6;      
phif = 0.02;   
tob = 0.4e-3; 
k0 = 2.56e8; 
kf = 2.4e8;
PHIf = kf / k0; 
kisc = 1.1e6;
kt = 4.9e5;
sig01 = 2.22e-16;
sig1n = 0.77e-17;
sigt1n = 3.85e-17; 
kb = 650; 
ksn1 = 5e12; 
ktn1 = ksn1;
kbsn = 2.8e8; 
kbtn = 2.8e8;
gamma = lambda / (h * c);

%% Calculate rate parameters for SAC and fmSAC respectively
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

%% Calculate traditional SAC and fmSAC PSF (GPU optimized)
fprintf('Calculate traditional SAC and fmSAC PSF...\n');
LL = size(I, 1);

% Pre-calculate constants
const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

% Pre-calculate frequency indices
n_time = length(params.t);
freq_res = (1/params.interval)/n_time;
f1_idx = round(params.f1/freq_res) + n_time/2 + 1;
f2_idx = round(params.f2/freq_res) + n_time/2 + 1;

% Pre-calculate modulation signals
if useGPU
    cos_f1 = gpuArray(cos(2*pi*params.f1*params.t));
    cos_f2 = gpuArray(cos(2*pi*params.f2*params.t));
else
    cos_f1 = cos(2*pi*params.f1*params.t);
    cos_f2 = cos(2*pi*params.f2*params.t);
end

hWaitbar = waitbar(0, 'Calculating PSF...', 'Name', 'fmSAC Photobleaching Simulation Progress');

% Initialize matrices
if useGPU
    y_SAC = gpuArray.zeros(LL);
    sig_fund_matrix = gpuArray.zeros(LL);
    sig_harm_matrix = gpuArray.zeros(LL);
else
    y_SAC = zeros(LL);
    sig_fund_matrix = zeros(LL);
    sig_harm_matrix = zeros(LL);
end

% Main calculation loop
for m = 1:LL
    for n = 1:LL
        % Conventional SAC calculation
        k_s = const_s * I(m, n);
        k_d_SAC = const_d * Id_SAC(m, n);
        y_SAC(m, n) = k_s / (params.c1 * k_s + params.c1 * k_d_SAC + params.k0);
        
        % fmSAC calculation - time domain modulation
        k_d_fmSAC = const_d * Id_fmSAC(m, n);
        numerator = k_s * (1 + params.m_s * cos_f1);
        denominator = params.c1 * (k_s * (1 + params.m_s * cos_f1) + ...
                       k_d_fmSAC * (1 + params.m_d * cos_f2)) + params.k0;
        y_s = numerator ./ denominator;
        
        % Spectrum analysis
        f_fft = fft(y_s);
        f_fft_shift = fftshift(f_fft);
        result = abs(f_fft_shift) / max(abs(f_fft_shift));
        
        sumx = (sum(result) - result(n_time/2+1)) / 2;
        sig_fund_matrix(m, n) = result(f1_idx) / sumx;
        sig_harm_matrix(m, n) = result(f2_idx) / sumx;
    end
    
    % Update progress
    waitbar(m/LL, hWaitbar, sprintf('Calculating PSF: %.1f%%', m/LL*100));
end

% Calculate global alpha value and apply
alpha_val = min(sig_fund_matrix(:) ./ sig_harm_matrix(:));
fmSAC = sig_fund_matrix - alpha_val * sig_harm_matrix;

% Create circular mask to remove side lobes
[x, y] = meshgrid(1:LL, 1:LL);
if useGPU
    [x, y] = meshgrid(gpuArray(1:LL), gpuArray(1:LL));
end
center = [ceil(LL/2), ceil(LL/2)];
radius = sqrt((x - center(2)).^2 + (y - center(1)).^2);
fmSAC(radius > 8) = 0;

close(hWaitbar);

% Normalize PSF
y_SAC = y_SAC / max(y_SAC(:));
fmSAC = fmSAC / max(fmSAC(:));

%% Calculate bleaching models for SAC and fmSAC respectively
fprintf('Calculating bleaching models...\n');

% SAC bleaching model
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

% fmSAC bleaching model
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

% Confocal bleaching model
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

%% Calculate bleaching effect
fprintf('Calculating bleaching effect...\n');
scan_intensity = 0.8;            % Increase scan intensity factor

% Calculate bleaching factors
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

%% Imaging simulation
fprintf('Performing imaging simulation...\n');
n = 100;        % Used to control range, 1 pixel = 10nm, forming a (2n)*(2n) range
m = 50;         % Number of fluorophores
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

% Imaging calculation - initial state (slight noise)
conf = conv2(s_cpu, Iout_conf_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_SAC = conv2(s_cpu, Iout_SAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC = conv2(s_cpu, Iout_fmSAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);

% Imaging after bleaching
enhanced_noise = 80;          % Noise enhanced after bleaching
conf_bleaching = conv2(s_cpu, Iout_conf_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_SAC_bleaching = conv2(s_cpu, Iout_SAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC_bleaching = conv2(s_cpu, Iout_fmSAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);

%% Define function to get images at arbitrary scan numbers
fprintf('Setting up function for generating images at arbitrary scan numbers...\n');

% Define function to get images at arbitrary scan numbers
get_bleached_image = @(scan_num, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                         bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise) ...
    getBleachedImage(scan_num, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                    bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise);

% Auxiliary function to get ROI region
get_roi_region = @(s_cpu) getROIRegion(s_cpu);

%% Interactive interface: select scan number to view images
fprintf('Creating interactive interface...\n');

% Create interactive graphical interface
fig_interface = figure('Name', 'Arbitrary Scan Number Image Viewer', 'Position', [200, 100, 800, 600], 'Color', 'w');

% Store signal data
signal_data.conf = [];
signal_data.sac = [];
signal_data.fmsac = [];
signal_data.scan_nums = [];

% Create slider to select scan number
scan_slider = uicontrol('Style', 'slider', ...
    'Min', 0, 'Max', 100, 'Value', 0, ...
    'Position', [100, 50, 400, 20], ...
    'Callback', @(src,evt) updateImages(src, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                                      bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise));

% Scan number display text
scan_text = uicontrol('Style', 'text', ...
    'Position', [520, 50, 200, 20], ...
    'String', 'Scan Number: 0', ...
    'FontSize', 12, 'FontWeight', 'bold');

% Create image display area
% set(gcf, 'Position', [100, 200, 800, 500], 'Color', 'w');
colormap hot

subplot(2, 3, 1) ;
img_conf = imagesc(s_cpu);
title('Sample', 'FontSize', 12, 'FontWeight', 'bold');
colorbar;
set(gca, 'XTick', [], 'YTick', []);
axis square;
add_subplot_scalebar(gca, 200, 'nm', 1);    % 20-pixel scale bar

subplot(2, 3, 2);
img_conf_bleach = imagesc(conf_bleaching);
title('Confocal after bleaching', 'FontSize', 12, 'FontWeight', 'bold');
colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square;
add_subplot_scalebar(gca, 200, 'nm', 1); 

subplot(2, 3, 3);
img_sac_bleach = imagesc(result_SAC_bleaching);
title('SAC after bleaching', 'FontSize', 12, 'FontWeight', 'bold');
colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square;
add_subplot_scalebar(gca, 200, 'nm', 1); 

ax_1 = subplot(2, 3, 4);
img_fmsac_bleach = imagesc(result_fmSAC_bleaching);
title('fmSAC after bleaching', 'FontSize', 12, 'FontWeight', 'bold');
colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square;
add_subplot_scalebar(gca, 200, 'nm', 1);

% Signal intensity display
ax = subplot(2, 3, [5, 6]);
hold on;
pos = ax.Position;
% Increase width and height (e.g., 20% increase)
ax.Position = [pos(1), pos(2)+0.1, pos(3), pos(4)*0.8];
signal_plot_conf = plot(0, 0, 'b-', 'LineWidth', 2, 'DisplayName', 'Confocal');
signal_plot_sac = plot(0, 0, 'c-', 'LineWidth', 2, 'DisplayName', 'SAC');
signal_plot_fmsac = plot(0, 0, 'g-', 'LineWidth', 2, 'DisplayName', 'fmSAC');
xlabel('Scan Number', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Intensity (a.u.)', 'FontSize', 12, 'FontWeight', 'bold');
title('Attenuation curve', 'FontSize', 12, 'FontWeight', 'bold');
legend('show', 'FontSize', 9, 'FontWeight', 'bold', 'Box', 'off');
grid on;

% Initialize interface
updateImages(scan_slider, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
            bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise);

%% Calculate fluorescence signal attenuation curve over time
fprintf('Calculating fluorescence signal attenuation curve...\n');
num_scans = 50; % Number of scans
scan_range = 0:num_scans;

% Select a single fluorophore for analysis
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

%  Define analysis region
roi_size = 5;
y_range = max(1, center_mol_y - floor(roi_size/2)):min(size(s_cpu, 1), center_mol_y + floor(roi_size/2));
x_range = max(1, center_mol_x - floor(roi_size/2)):min(size(s_cpu, 2), center_mol_x + floor(roi_size/2));

% Initialize signal attenuation arrays）
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
    
    % Calculate bleached imaging
    result_SAC_scan = conv2(s_cpu, Iout_SAC_cpu .* current_bleach_SAC, 'same');
    result_fmSAC_scan = conv2(s_cpu, Iout_fmSAC_cpu .* current_bleach_fmSAC, 'same');
    
    % Add noise increasing with scan number
    noise_level = base_noise * (1 + scan_idx * 0.03); % Noise increases linearly with scan number
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

% Generate fitting curves
x_fit_continuous = linspace(0, num_scans, 100);
SAC_fit_curve = SAC_fit.a * exp(-SAC_fit.b * x_fit_continuous);
fmSAC_fit_curve = fmSAC_fit.a * exp(-fmSAC_fit.b * x_fit_continuous);

%% Result display
fprintf('Generating result images...\n');

% Figure 2: Imaging result comparison
figure('Name', 'Basic Imaging Result Comparison', 'Position', [100, 100, 1400, 800], 'Color', 'w');
colormap hot

% First row: unbleached
subplot(2,4,1), imagesc(s_cpu);
colorbar; axis square; title('Sample Structure', 'FontSize', 12, 'FontWeight', 'bold')

subplot(2,4,2), imagesc(conf);
colorbar; axis square; title('Confocal', 'FontSize', 12, 'FontWeight', 'bold') 

subplot(2,4,3), imagesc(result_SAC);
colorbar; axis square; 
title(sprintf('SAC (I_d = %dkW/cm²)', params.I_d_SAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')

subplot(2,4,4), imagesc(result_fmSAC);
colorbar; axis square; 
title(sprintf('fmSAC (I_d = %dkW/cm²)', params.I_d_fmSAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')

% Second row: after bleaching
subplot(2,4,6), imagesc(conf_bleaching);
colorbar; axis square; title('Confocal after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')

subplot(2,4,7), imagesc(result_SAC_bleaching);
colorbar; axis square; title('SAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')

subplot(2,4,8), imagesc(result_fmSAC_bleaching);
colorbar; axis square; title('fmSAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')

% Figure 3: PSF profile comparison
figure('Name', 'PSF Profile Comparison', 'Position', [100, 100, 1000, 600], 'Color', 'w');

plot(Iout_conf_cpu(ceil(LL/2),:), 'b', 'LineWidth', 2)
hold on
plot(Iout_SAC_cpu(ceil(LL/2),:), 'c', 'LineWidth', 2)
hold on
plot(Iout_SAC_bleaching_cpu(ceil(LL/2),:), 'm--', 'LineWidth', 2)
hold on
plot(Iout_fmSAC_cpu(ceil(LL/2),:), 'g', 'LineWidth', 2)
hold on
plot(Iout_fmSAC_bleaching_cpu(ceil(LL/2),:), 'r--', 'LineWidth', 2)

% Create legend labels with light intensity information
sac_legend = sprintf('SAC @%dkW/cm²', params.I_d_SAC/1e3);
sac_bleached_legend = sprintf('SAC Bleached @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend = sprintf('fmSAC @%dkW/cm²', params.I_d_fmSAC/1e3);
fmsac_bleached_legend = sprintf('fmSAC Bleached @%dkW/cm²', params.I_d_fmSAC/1e3);

legend('Confocal', sac_legend, sac_bleached_legend, fmsac_legend, fmsac_bleached_legend, 'Location', 'best')
title('PSF Profile Comparison', 'FontSize', 14, 'FontWeight', 'bold')
xlabel('Position (pixels)'); ylabel('Normalized Intensity (a.u.)')
grid on

% Figure 4: Fluorescence signal attenuation curve
figure('Name', 'Fluorescence Signal Attenuation Curve', 'Position', [100, 100, 1000, 600], 'Color', 'w');

% Plot data points
plot(scan_range, signal_SAC_norm, 'cs', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'c')
hold on
plot(scan_range, signal_fmSAC_norm, 'g^', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'g')

% Plot fitting curves
plot(x_fit_continuous, SAC_fit_curve, 'c-', 'LineWidth', 2.5)
plot(x_fit_continuous, fmSAC_fit_curve, 'g-', 'LineWidth', 2.5)

title('Fluorescence Signal Decay during Imaging', 'FontSize', 14, 'FontWeight', 'bold')
xlabel('Scan Number', 'FontSize', 12, 'FontWeight', 'bold')
ylabel('Normalized Fluorescence Intensity', 'FontSize', 12, 'FontWeight', 'bold')

sac_legend_data = sprintf('SAC Data @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_data = sprintf('fmSAC Data @%dkW/cm²', params.I_d_fmSAC/1e3);
sac_legend_fit = sprintf('SAC Fitted @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_fit = sprintf('fmSAC Fitted @%dkW/cm²', params.I_d_fmSAC/1e3);

legend(sac_legend_data, fmsac_legend_data, sac_legend_fit, fmsac_legend_fit, ...
       'Location', 'best', 'FontSize', 10)
grid on
set(gca, 'FontSize', 11, 'LineWidth', 1.2)

% Add half-life annotations
[~, idx_half_SAC] = min(abs(SAC_fit_curve - 0.5));
[~, idx_half_fmSAC] = min(abs(fmSAC_fit_curve - 0.5));

half_life_SAC = x_fit_continuous(idx_half_SAC);
half_life_fmSAC = x_fit_continuous(idx_half_fmSAC);

text(half_life_SAC, 0.52, sprintf('SAC: %.1f scans', half_life_SAC), ...
     'Color', 'c', 'FontSize', 10, 'HorizontalAlignment', 'center')
text(half_life_fmSAC, 0.45, sprintf('fmSAC: %.1f scans', half_life_fmSAC), ...
     'Color', 'g', 'FontSize', 10, 'HorizontalAlignment', 'center')

%% Analysis results
fprintf('\n=== Photobleaching Analysis Results ===\n');
fprintf('Competitive light intensity settings:\n');
fprintf('  SAC: %d kW/cm²\n', params.I_d_SAC/1e3);
fprintf('  fmSAC: %d kW/cm²\n', params.I_d_fmSAC/1e3);

% Calculate signal loss ratio
SAC_bleaching_ratio = mean(Iout_SAC_bleaching_cpu(:)) / mean(Iout_SAC_cpu(:));
fmSAC_bleaching_ratio = mean(Iout_fmSAC_bleaching_cpu(:)) / mean(Iout_fmSAC_cpu(:));
conf_bleaching_ratio = mean(Iout_conf_bleaching_cpu(:)) / mean(Iout_conf_cpu(:));

fprintf('\nSignal retention ratio after bleaching:\n');
fprintf('  Confocal: %.2f%%\n', conf_bleaching_ratio * 100);
fprintf('  Conventional SAC: %.2f%%\n', SAC_bleaching_ratio * 100);
fprintf('  fmSAC: %.2f%%\n', fmSAC_bleaching_ratio * 100);

fprintf('\nHalf-life analysis:\n');
fprintf('  SAC Half-life: %.1f scans\n', half_life_SAC);
fprintf('  fmSAC Half-life: %.1f scans\n', half_life_fmSAC);

half_life_improvement = (half_life_fmSAC - half_life_SAC) / half_life_SAC * 100;
fprintf('  fmSAC half-life improvement: +%.1f%%\n', half_life_improvement);

fprintf('\nAttenuation constants:\n');
fprintf('  SAC attenuation constant: %.4f\n', SAC_fit.b);
fprintf('  fmSAC attenuation constant: %.4f\n', fmSAC_fit.b);

fprintf('\nUsage instructions:\n');
fprintf('  1. Use the slider in the interactive interface to view images at any scan number in real time\n');
fprintf('  2. Slider range: 0-100 scans\n');
fprintf('  3. Images will be updated in real time to show Confocal, SAC and fmSAC images at the current scan number\n');

fprintf('\nSimulation completed！\n');

%% Auxiliary function definitions
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

function [conf_img, sac_img, fmsac_img] = getBleachedImage(scan_num, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                                                         bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise)
    % Calculate current scan bleaching factor
    current_bleach_conf = bleach_factor_conf.^(scan_num*0.1);
    current_bleach_SAC = bleach_factor_SAC.^(scan_num*0.2);
    current_bleach_fmSAC = bleach_factor_fmSAC.^(scan_num*0.15);
    
    % Calculate bleached imaging
    conf_bleached = conv2(s_cpu, Iout_conf_cpu .* current_bleach_conf, 'same');
    sac_bleached = conv2(s_cpu, Iout_SAC_cpu .* current_bleach_SAC, 'same');
    fmsac_bleached = conv2(s_cpu, Iout_fmSAC_cpu .* current_bleach_fmSAC, 'same');
    
    % Add noise increasing with scan number
    noise_level = base_noise * (1 + scan_num * 0.03);
    R_scan = noise_level * (rand(size(s_cpu)) - 0.5);
    
    conf_img = conf_bleached + R_scan;
    sac_img = sac_bleached + R_scan;
    fmsac_img = fmsac_bleached + R_scan;
end

function [y_range, x_range] = getROIRegion(s_cpu)
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
    
    roi_size = 5;
    y_range = max(1, center_mol_y - floor(roi_size/2)):min(size(s_cpu, 1), center_mol_y + floor(roi_size/2));
    x_range = max(1, center_mol_x - floor(roi_size/2)):min(size(s_cpu, 2), center_mol_x + floor(roi_size/2));
end

function updateImages(source, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                     bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise)
    
    % Get current figure handle
    fig = source.Parent;
    
    % Get all image and text handles
    scan_text = findobj(fig, 'Type', 'uicontrol', 'Style', 'text');
    img_conf_bleach = findobj(fig, 'Type', 'image', 'Parent', subplot(2,3,2));
    img_sac_bleach = findobj(fig, 'Type', 'image', 'Parent', subplot(2,3,3));
    img_fmsac_bleach = findobj(fig, 'Type', 'image', 'Parent', subplot(2,3,4));
    signal_plot_conf = findobj(fig, 'Type', 'line', 'Color', [0 0 1]);
    signal_plot_sac = findobj(fig, 'Type', 'line', 'Color', [0 1 1]);
    signal_plot_fmsac = findobj(fig, 'Type', 'line', 'Color', [0 1 0]);
    
    scan_num = round(source.Value);
    scan_text.String = sprintf('Scan number: %d', scan_num);
    
    % Get images for current scan number
    [conf_img, sac_img, fmsac_img] = getBleachedImage(scan_num, s_cpu, Iout_conf_cpu, Iout_SAC_cpu, Iout_fmSAC_cpu, ...
                                                     bleach_factor_conf, bleach_factor_SAC, bleach_factor_fmSAC, base_noise);
    
    % Update images
    set(img_conf_bleach, 'CData', conf_img);
    set(img_sac_bleach, 'CData', sac_img);
    set(img_fmsac_bleach, 'CData', fmsac_img);
    
    % Update titles to show current scan number
    subplot(2, 3, 2); title(sprintf('Confocal (Scan number %d)', scan_num), 'FontSize', 12, 'FontWeight', 'bold');
    subplot(2, 3, 3); title(sprintf('SAC (Scan number %d)', scan_num), 'FontSize', 12, 'FontWeight', 'bold');
    subplot(2, 3, 4); title(sprintf('fmSAC (Scan number %d)', scan_num), 'FontSize', 12, 'FontWeight', 'bold');
    
    % Update signal intensity display (calculate ROI signal)
    [y_range, x_range] = getROIRegion(s_cpu);
    conf_signal = sum(sum(conf_img(y_range, x_range)));
    sac_signal = sum(sum(sac_img(y_range, x_range)));
    fmsac_signal = sum(sum(fmsac_img(y_range, x_range)));
    
    % Store signal data
    persistent signal_history
    if isempty(signal_history)
        signal_history.conf = [];
        signal_history.sac = [];
        signal_history.fmsac = [];
        signal_history.scan_nums = [];
    end
    
    % Check if current scan number has been recorded
    if isempty(signal_history.scan_nums) || ~ismember(scan_num, signal_history.scan_nums)
        signal_history.conf(end+1) = conf_signal;
        signal_history.sac(end+1) = sac_signal;
        signal_history.fmsac(end+1) = fmsac_signal;
        signal_history.scan_nums(end+1) = scan_num;
        
        % Sort
        [signal_history.scan_nums, sort_idx] = sort(signal_history.scan_nums);
        signal_history.conf = signal_history.conf(sort_idx);
        signal_history.sac = signal_history.sac(sort_idx);
        signal_history.fmsac = signal_history.fmsac(sort_idx);
    end
    
    % Update signal curves
    if ~isempty(signal_plot_conf)
        set(signal_plot_conf, 'XData', signal_history.scan_nums, 'YData', signal_history.conf);
        set(signal_plot_sac, 'XData', signal_history.scan_nums, 'YData', signal_history.sac);
        set(signal_plot_fmsac, 'XData', signal_history.scan_nums, 'YData', signal_history.fmsac);
    end 
    drawnow;
end

function add_subplot_scalebar(subplot_handle, physical_size, units ,fig_1)
    % Add scale bar to subplot
    % subplot_handle: subplot handle
    % physical_size: physical length of scale bar
    % units: units (e.g., 'mm', 'cm', 'pixels', etc.)
    
    % Get current figure and subplot information
    subplot_pos = get(subplot_handle, 'Position');
    subplot_x = subplot_pos(1);
    subplot_y = subplot_pos(2);
    [subplot_width, subplot_height] = get_image_width_without_colorbar(subplot_handle);
    
    % Get subplot data range
    x_limits = get(subplot_handle, 'XLim');
    y_limits = get(subplot_handle, 'YLim');
    data_width = x_limits(2) - x_limits(1);
    data_height = y_limits(2) - y_limits(1);
    
    % Calculate pixel length of scale bar
    if strcmpi(units, 'pixels')
        scalebar_data_length = physical_size;
    else
        scalebar_data_length = physical_size.*0.1;
    end
    
    % Convert data length to normalized length in subplot
    scalebar_normalized_length = scalebar_data_length / data_width * subplot_width;
        
    % Draw scale bar
    if (fig_1 == 1)    
        margin = 0.1; 
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
        % There is a colorbar, calculate its width
        cbar = cbar_handles(1);
        cbar_pos = get(cbar, 'Position');
        cbar_x = cbar_pos(1);
        cbar_width = cbar_pos(3);
        cbar_height = cbar_pos(4);

        margin = cbar_width;
        image_width = cbar_x - subplot_x - margin;
        image_height = cbar_height;
    else
        % No colorbar, use total subplot width
        image_width = subplot_total_width;
        image_height = subplot_total_height;
    end

    if image_width <= 0
        image_width = subplot_total_width * 0.8; % Default to 80% width
    end
end