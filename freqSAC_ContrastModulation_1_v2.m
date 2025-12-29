%% 探讨调制对比度对fmSAC的FWHM的影响 - GPU加速版本
% X轴为I_exc，Y轴为fwhm，CM=[0.1,0.3,0.5,0.7,0.9]
% 添加进度条，自动适应不同数据大小
% GPU加速版本

clc
clear all
close all
addpath(genpath('PSF'));
addpath(genpath('CSV'));
tic;

% 检查GPU可用性
if gpuDeviceCount > 0
    useGPU = true;
    gpu = gpuDevice();
    fprintf('使用GPU加速: %s\n', gpu.Name);
else
    useGPU = false;
    fprintf('未检测到GPU，使用CPU计算\n');
end

%% 物理常数和参数设置
k_isc = 1.1e6;
k_t = 0.49e6;
k0 = 2.56e8;
c1 = 1 + k_isc/k_t;
h = 6.626e-34;      % 普兰克常数 
c = 3e10;           % 光速，以cm做度量
lambda_s = 532e-7;  % 以cm做度量
lambda_d = 488e-7;

sigma_s = 2.7e-16;  % 激发光吸收截面(针对532nm波长)
sigma_d = sigma_s * 0.512063188; % 采用488nm波长

% I_s = 10e3;         % 对应光强10k W/cm2
I_d = 10e3;         % 对应光强500k W/cm2
f1 = 10e3;          % f1频率
f2 = 15e3;          % f2频率
interval = 10e-6;
t = 0:interval:1-interval;
m_d = 1.0;            % 竞争调制对比度CM=(Imax-Imin)/(Imax+Imin)

% 将时间向量转移到GPU（如果可用）
if useGPU
    t_gpu = gpuArray(t);
else
    t_gpu = t;
end

%% 加载PSF数据
% a=load('I_exc532_51_3D.mat');
% I1=a.result.PSF(:,:,25);
% b=load('I_hexc488_51_3D.mat');
% I2=b.result.PSF(:,:,25);


a=load('I_exc532_501.mat');
I1=a.result.PSF;
b=load('I_hexc488_501.mat');
I2=b.result.PSF;
% 归一化并缩放
I1 = I1 / max(I1(:));
I2 = I2 / max(I2(:));
% I_exc = I_s * I1;   % 对应光强10kW/cm2
I_hexc = I_d* I2;   % 对应光强10kW/cm2

[rows, cols] = size(I1);
center_col = round(cols/2); % 自动确定中心列

%% 初始化参数
I_s_values = (1:10) * 5e3; % I_s值数组
modulation_depths = 0.1:0.1:1.0; % 调制深度数组
num_modulations = length(modulation_depths);
num_I_s= length(I_s_values);

% 预分配结果矩阵
data = zeros(num_modulations + 1, num_I_s); % +1 用于传统SAC
FWHM_temp = zeros(1, num_I_s); 

%% FFT相关参数预计算
N = length(t);
frequencies = (-N/2:N/2-1) * (1/(N*interval));
f1_idx = find(abs(frequencies - f1) == min(abs(frequencies - f1)), 1);
f2_idx = find(abs(frequencies - f2) == min(abs(frequencies - f2)), 1);

% 将频率索引转移到GPU（如果可用）
if useGPU
    f1_idx_gpu = gpuArray(f1_idx);
    f2_idx_gpu = gpuArray(f2_idx);
    N_gpu = gpuArray(N);
else
    f1_idx_gpu = f1_idx;
    f2_idx_gpu = f2_idx;
    N_gpu = N;
end

%% 计算传统SAC的FWHM
fprintf('计算传统SAC...\n');
progressBar = waitbar(0, '计算传统SAC: 0%', 'Name', '进度');

for m = 1:num_I_s
    I_exc = I_s_values(m) * I1;
    y_SAC = zeros(rows, 1);
    
    for i = 1:rows
        k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
        k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
        y_SAC(i) = k_s / (c1 * k_s + k0 + c1 * k_d);
    end
    
    y_SAC = y_SAC / max(y_SAC);
    half_max = max(y_SAC) / 2;
    half_index = find(y_SAC >= half_max);
    FWHM_temp(m) = length(half_index); % 500像素代表500nm，一个像素代表1nm
    
    waitbar(m/num_I_s, progressBar, sprintf('计算传统SAC: %.0f%%', m/num_I_s*100));
end

data(1, :) = FWHM_temp;
close(progressBar);

%% 计算fmSAC的FWHM - GPU加速版本
fprintf('计算fmSAC...\n');
total_iterations = num_modulations * num_I_s;
progressBar = waitbar(0, '计算fmSAC: 0%', 'Name', '进度');
iteration_count = 0;

for n = 1:num_modulations
    m_s = modulation_depths(n);
    
    for m = 1:num_I_s
        I_exc = I_s_values(m) * I1;
        sig_fund = zeros(rows, 1);
        sig_harm = zeros(rows, 1);
        
        for i = 1:rows
            % 计算速率常数
            k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
            k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
            
            % 双调制SAC信号 - 使用GPU计算
            if useGPU
                % 在GPU上计算
                k_s_gpu = gpuArray(k_s);
                k_d_gpu = gpuArray(k_d);
                m_s_gpu = gpuArray(m_s);
                m_d_gpu = gpuArray(m_d);
                k0_gpu = gpuArray(k0);
                c1_gpu = gpuArray(c1);
                
                % GPU计算
                y_s_gpu = (k_s_gpu * (1 + m_s_gpu * cos(2*pi*f1*t_gpu))) ./ ...
                          (c1_gpu * (k_s_gpu * (1 + m_s_gpu * cos(2*pi*f1*t_gpu)) + ...
                           k_d_gpu * (1 + m_d_gpu * cos(2*pi*f2*t_gpu))) + k0_gpu);
                
                % GPU FFT分析
                f_omiga_gpu = fft(y_s_gpu);
                f_omiga_shift_gpu = fftshift(f_omiga_gpu);
                result_gpu = abs(f_omiga_shift_gpu) / max(abs(f_omiga_shift_gpu));
                
                % 计算总功率（去除DC分量）
                total_power_gpu = sum(result_gpu) - result_gpu(N_gpu/2+1);
                
                % 提取特定频率分量
                sig_fund(i) = gather(result_gpu(f1_idx_gpu) / total_power_gpu);
                sig_harm(i) = gather(result_gpu(f2_idx_gpu) / total_power_gpu);
            else
                % CPU计算（原代码）
                y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
                      (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);
                
                % FFT分析
                f_omiga = fft(y_s);
                f_omiga_shift = fftshift(f_omiga);
                result = abs(f_omiga_shift) / max(abs(f_omiga_shift));
                
                % 计算总功率（去除DC分量）
                total_power = sum(result) - result(N/2+1);
                
                % 提取特定频率分量
                sig_fund(i) = result(f1_idx) / total_power;
                sig_harm(i) = result(f2_idx) / total_power;
            end
        end
        
        % 计算fmSAC信号
        alpha = min(sig_fund ./ sig_harm);
        fmSAC_signal = sig_fund - alpha * sig_harm;
        
        % 归一化并计算FWHM
        fmSAC_signal = fmSAC_signal / max(fmSAC_signal);
        half_max = max(fmSAC_signal) / 2;
        half_index = find(fmSAC_signal >= half_max);
        FWHM_temp(m) = length(half_index);
        
        % 更新进度条
        iteration_count = iteration_count + 1; 
        waitbar(iteration_count/total_iterations, progressBar, ...
            sprintf('计算fmSAC (CM=%.1f): %.0f%%', m_d, iteration_count/total_iterations*100));
    end
    
    data(n+1, :) = FWHM_temp;
end

close(progressBar);

%% 绘图
figure('Position', [100, 100, 1400, 750]);
hold on;

colors = {'k','r','g','b','c','m','#EDB120','#4DBEEE','y','#7E2F8E','#77AC30'};
line_styles = {'-','--',':','-.','--',':','-.','--',':','-.','--'};
markers = {'s','|','d','^','v','p','h','+','o','.','*'};

% 根据数据大小自动调整线宽和标记大小
if rows > 1000
    line_width = 1.5;
    marker_size = 6;
else
    line_width = 2;
    marker_size = 8;
end

% 绘制所有曲线
legend_labels = cell(num_modulations + 1, 1);
legend_labels{1} = 'SAC';

for j = 1:num_modulations + 1
    if j == 1
        % 传统SAC
        plot(I_s_values/1000, data(j, :), ...
            'Color', colors{1}, ...
            'LineStyle', line_styles{1}, ...
            'LineWidth', line_width, ...
            'Marker', markers{1}, ...
            'MarkerSize', marker_size);
    else
        % fmSAC
        plot(I_s_values/1000, data(j, :), ...
            'Color', colors{mod(j-1, length(colors)) + 1}, ...
            'LineStyle', line_styles{mod(j-1, length(line_styles)) + 1}, ...
            'LineWidth', line_width, ...
            'Marker', markers{mod(j-1, length(markers)) + 1}, ...
            'MarkerSize', marker_size);
        
        legend_labels{j} = sprintf('fmSAC CM_{1}=%.1f', modulation_depths(j-1));
    end
end

%% 图形美化
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('FWHM (nm)', 'FontWeight', 'bold', 'FontSize', 24);
xlabel('I_{exc} (kW/cm^{2})', 'FontWeight', 'bold', 'FontSize', 24);
% title('调制对比度对fmSAC的FWHM的影响 (GPU加速)', 'FontWeight', 'bold', 'FontSize', 18);

grid on;
box on;

% 根据数据范围自动调整坐标轴
xlim([min(I_s_values/1000), max(I_s_values/1000)]);
ylim([120, max(data(:)) * 1.05+20]);

legend(legend_labels, 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12, ...
    'Location', 'northwest');

% set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);


fprintf('计算完成！\n');
elapsedTime = toc;
fprintf('代码运行时间为: %.4f 秒\n', elapsedTime);

% 显示GPU内存使用情况（如果使用GPU）- 兼容不同MATLAB版本
if useGPU
    try
        % 尝试新版本的属性名
        gpuInfo = gpuDevice();
        if isprop(gpuInfo, 'UsedMemory') && isprop(gpuInfo, 'AvailableMemory')
            fprintf('GPU内存使用: %.2f MB / %.2f MB\n', ...
                gpuInfo.UsedMemory/1e6, gpuInfo.AvailableMemory/1e6);
        elseif isprop(gpuInfo, 'TotalMemory')
            % 旧版本MATLAB
            fprintf('GPU总内存: %.2f MB\n', gpuInfo.TotalMemory/1e6);
        else
            fprintf('GPU信息: %s\n', gpuInfo.Name);
        end
    catch
        fprintf('无法获取GPU内存信息\n');
    end
end