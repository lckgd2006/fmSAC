%% Explore the effect of modulation contrast on FWHM of fmSAC - GPU accelerated version
% X-axis: I_hexc, Y-axis: fwhm, CM=[0.1,0.3,0.5,0.7,0.9]
% Add progress bar, automatically adapt to different data sizes
% GPU accelerated version

clc
clear all
close all
addpath(genpath('PSF'));
addpath(genpath('CSV'));
tic;

% Check GPU availability
if gpuDeviceCount > 0
    useGPU = true;
    gpu = gpuDevice();
    fprintf('Using GPU acceleration: %s\n', gpu.Name);
else
    useGPU = false;
    fprintf('No GPU detected, using CPU for computation\n');
end

%% Physical constants and parameter settings
k_isc = 1.1e6;
k_t = 0.49e6;
k0 = 2.56e8;
c1 = 1 + k_isc/k_t;
h = 6.626e-34;      
c = 3e10;           
lambda_s = 532e-7;  
lambda_d = 488e-7;
sigma_s = 2.7e-16;  
sigma_d = sigma_s * 0.512063188; 
I_s = 10e3;         
f1 = 10e3;         
f2 = 15e3;         
interval = 10e-6;
t = 0:interval:1-interval;
m_s = 0.1;           

% Transfer time vector to GPU if available
if useGPU
    t_gpu = gpuArray(t);
else
    t_gpu = t;
end

%% Load PSF data
a=load('I_exc532_501.mat');
I1=a.result.PSF;
b=load('I_hexc488_501.mat');
I2=b.result.PSF;

% Normalization and scaling
I1 = I1 / max(I1(:));
I2 = I2 / max(I2(:));
I_exc = I_s * I1;           % Excitation light intensity (10kW/cm2)
[rows, cols] = size(I1);
center_col = round(cols/2); % Automatically determine central column

%% Initialize parameters
I_d_values = (1:10) * 50e3;                  % I_d value array
modulation_depths = 0.1:0.1:1.0;             % Modulation depth array
num_modulations = length(modulation_depths);
num_I_d = length(I_d_values);

% Preallocate result matrix
data = zeros(num_modulations + 1, num_I_d); 
FWHM_temp = zeros(1, num_I_d); 

%% Precompute FFT-related parameters
N = length(t);
frequencies = (-N/2:N/2-1) * (1/(N*interval));
f1_idx = find(abs(frequencies - f1) == min(abs(frequencies - f1)), 1);
f2_idx = find(abs(frequencies - f2) == min(abs(frequencies - f2)), 1);

% Transfer frequency indices to GPU if available
if useGPU
    f1_idx_gpu = gpuArray(f1_idx);
    f2_idx_gpu = gpuArray(f2_idx);
    N_gpu = gpuArray(N);
else
    f1_idx_gpu = f1_idx;
    f2_idx_gpu = f2_idx;
    N_gpu = N;
end

%% Calculate FWHM of conventional SAC
fprintf('Calculating conventional SAC...\n');
progressBar = waitbar(0, 'Calculating conventional SAC: 0%', 'Name', 'Progress');

for m = 1:num_I_d
    I_hexc = I_d_values(m) * I2;
    y_SAC = zeros(rows, 1);
    
    for i = 1:rows
        k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
        k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
        y_SAC(i) = k_s / (c1 * k_s + k0 + c1 * k_d);
    end
    
    y_SAC = y_SAC / max(y_SAC);
    half_max = max(y_SAC) / 2;
    half_index = find(y_SAC >= half_max);
    FWHM_temp(m) = length(half_index);     % 500 pixels = 500nm, 1 pixel = 1nm
    
    waitbar(m/num_I_d, progressBar, sprintf('Calculating conventional SAC: %.0f%%', m/num_I_d*100));
end

data(1, :) = FWHM_temp;
close(progressBar);

%% Calculate FWHM of fmSAC - GPU accelerated version
fprintf('Calculating fmSAC...\n');
total_iterations = num_modulations * num_I_d;
progressBar = waitbar(0, 'Calculating fmSAC: 0%', 'Name', 'Progress');
iteration_count = 0;

for n = 1:num_modulations
    m_d = modulation_depths(n);
    
    for m = 1:num_I_d
        I_hexc = I_d_values(m) * I2;
        sig_fund = zeros(rows, 1);
        sig_harm = zeros(rows, 1);
        
        for i = 1:rows
            % Calculate rate constants
            k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
            k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
            
            % Dual-modulation SAC signal - GPU calculation
            if useGPU
                % Calculate SAC signal on GPU 
                k_s_gpu = gpuArray(k_s);
                k_d_gpu = gpuArray(k_d);
                m_s_gpu = gpuArray(m_s);
                m_d_gpu = gpuArray(m_d);
                k0_gpu = gpuArray(k0);
                c1_gpu = gpuArray(c1);
                
                y_s_gpu = (k_s_gpu * (1 + m_s_gpu * cos(2*pi*f1*t_gpu))) ./ ...
                          (c1_gpu * (k_s_gpu * (1 + m_s_gpu * cos(2*pi*f1*t_gpu)) + ...
                           k_d_gpu * (1 + m_d_gpu * cos(2*pi*f2*t_gpu))) + k0_gpu);
                
                % FFT analysis
                f_omiga_gpu = fft(y_s_gpu);
                f_omiga_shift_gpu = fftshift(f_omiga_gpu);
                result_gpu = abs(f_omiga_shift_gpu) / max(abs(f_omiga_shift_gpu));
                
                % Calculate total power
                total_power_gpu = sum(result_gpu) - result_gpu(N_gpu/2+1);
                
                % Extract specific frequency components
                sig_fund(i) = gather(result_gpu(f1_idx_gpu) / total_power_gpu);
                sig_harm(i) = gather(result_gpu(f2_idx_gpu) / total_power_gpu);
            else
                % Calculate SAC signal on CPU 
                y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
                      (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);
                
                % FFT analysis
                f_omiga = fft(y_s);
                f_omiga_shift = fftshift(f_omiga);
                result = abs(f_omiga_shift) / max(abs(f_omiga_shift));
                
                % Calculate total power
                total_power = sum(result) - result(N/2+1);
                
                % Extract specific frequency components
                sig_fund(i) = result(f1_idx) / total_power;
                sig_harm(i) = result(f2_idx) / total_power;
            end
        end
        
        % Calculate fmSAC signal
        alpha = min(sig_fund ./ sig_harm);
        fmSAC_signal = sig_fund - alpha * sig_harm;
        
        % Normalization and FWHM calculation
        fmSAC_signal = fmSAC_signal / max(fmSAC_signal);
        half_max = max(fmSAC_signal) / 2;
        half_index = find(fmSAC_signal >= half_max);
        FWHM_temp(m) = length(half_index);
        
        % Update progress bar
        iteration_count = iteration_count + 1; 
        waitbar(iteration_count/total_iterations, progressBar, ...
            sprintf('Calculating fmSAC (CM=%.1f): %.0f%%', m_d, iteration_count/total_iterations*100));
    end   
    data(n+1, :) = FWHM_temp;
end

close(progressBar);

%% Plotting
figure('Position', [100, 100, 1400, 750]);
hold on;

colors = {'k','r','g','b','c','m','#EDB120','#4DBEEE','y','#7E2F8E','#77AC30'};
line_styles = {'-','--',':','-.','--',':','-.','--',':','-.','--'};
markers = {'s','|','d','^','v','p','h','+','o','.','*'};

% Automatically adjust line width and marker size based on data size
if rows > 1000
    line_width = 1.5;
    marker_size = 6;
else
    line_width = 2;
    marker_size = 8;
end

% Plot all curves
legend_labels = cell(num_modulations + 1, 1);
legend_labels{1} = 'SAC';

for j = 1:num_modulations + 1
    if j == 1
        % Conventional SAC
        plot(I_d_values/1000, data(j, :), ...
            'Color', colors{1}, ...
            'LineStyle', line_styles{1}, ...
            'LineWidth', line_width, ...
            'Marker', markers{1}, ...
            'MarkerSize', marker_size);
    else
        % fmSAC
        plot(I_d_values/1000, data(j, :), ...
            'Color', colors{mod(j-1, length(colors)) + 1}, ...
            'LineStyle', line_styles{mod(j-1, length(line_styles)) + 1}, ...
            'LineWidth', line_width, ...
            'Marker', markers{mod(j-1, length(markers)) + 1}, ...
            'MarkerSize', marker_size);
        
        legend_labels{j} = sprintf('fmSAC CM_{2}=%.1f', modulation_depths(j-1));
    end
end

%% Figure beautification
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('FWHM (nm)', 'FontWeight', 'bold', 'FontSize', 24);
xlabel('I_{hexc} (kW/cm^{2})', 'FontWeight', 'bold', 'FontSize', 24);
% title('Effect of modulation contrast on FWHM of fmSAC (GPU accelerated)', 'FontWeight', 'bold', 'FontSize', 18);

grid on;
box on;

% Automatically adjust axes based on data range
xlim([min(I_d_values/1000), max(I_d_values/1000)]);
ylim([20, max(data(:)) * 1.05]);

legend(legend_labels, 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12, ...
    'Location', 'best');

% set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);


fprintf('Calculation completed!\n');
elapsedTime = toc;
fprintf('Total runtime: %.4f 秒\n', elapsedTime);

% Display GPU memory usage (if GPU is used) - compatible with different MATLAB versions
if useGPU
    try
        % Try new version property names
        gpuInfo = gpuDevice();
        if isprop(gpuInfo, 'UsedMemory') && isprop(gpuInfo, 'AvailableMemory')
            fprintf('GPU Memory Usage: %.2f MB / %.2f MB\n', ...
                gpuInfo.UsedMemory/1e6, gpuInfo.AvailableMemory/1e6);
        elseif isprop(gpuInfo, 'TotalMemory')
            % For older MATLAB versions
            fprintf('GPU Total Memory: %.2f MB\n', gpuInfo.TotalMemory/1e6);
        else
            fprintf('GPU Information: %s\n', gpuInfo.Name);
        end
    catch
        fprintf('Unable to retrieve GPU memory information\n');
    end
end