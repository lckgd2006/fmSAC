%% 做fmSAC与fmSAC+的对比
% 探讨调制对比度m_d对fmSAC的影响
% 仅计算中心行数据优化版本 - 结构体参数版本
clc; clear; close all;
addpath(genpath('PSF'));
addpath(genpath('CSV'));

%% 参数初始化 - 使用结构体组织
params = struct();

% 物理常数
params.h = 6.626e-34;        % 普朗克常数
params.c = 3e10;             % 光速，以cm做度量
params.lambda_s = 532e-7;    % 激发波长，以cm做度量
params.lambda_d = 488e-7;    % 竞争波长，以cm做度量

% 速率常数
params.k_isc = 1.1e6;
params.k_t = 0.49e6;
params.k0 = 2.56e8;
params.c1 = 1 + params.k_isc/params.k_t;

% 吸收截面
params.sigma_s = 2.7e-16;    % 激发光吸收截面(针对532nm波长)
params.sigma_d = params.sigma_s * 0.512063188; % 488nm波长吸收截面

% 光强参数
params.I_s = 10e3;           % 对应光强10k W/cm²
params.I_d = 500e3;          % 对应光强500 kW/cm²

% 频率参数
params.f1 = 10e3;            % f1频率
params.f2 = 15e3;            % f2频率

% 时间参数
params.interval = 10e-6;
params.t = 0:params.interval:1-params.interval;
params.n_time = length(params.t);

% 调制对比度参数
params.m_d = 1.0;            % 竞争调制对比度
params.m_s_1 = 0.3;          % 激发光1调制对比度
params.m_s_2 = 0.6;          % 激发光2调制对比度
params.m_s_3 = 0.9;          % 激发光3调制对比度
params.m_s_4 = 0.1;          % 激发光4调制对比度

% 计算相关常数
params.const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
params.const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

% 频率索引
params.freq_res = (1/params.interval)/params.n_time;
params.f1_idx = round(params.f1/params.freq_res) + floor(params.n_time/2) + 1;
params.f2_idx = round(params.f2/params.freq_res) + floor(params.n_time/2) + 1;
params.f1_idx = min(max(params.f1_idx, 1), params.n_time);
params.f2_idx = min(max(params.f2_idx, 1), params.n_time);
params.center_freq_idx = floor(params.n_time/2) + 1;

%% 检查GPU可用性
if gpuDeviceCount > 0
    params.useGPU = true;
    gpu = gpuDevice();
    fprintf('使用GPU: %s\n', gpu.Name);
else
    params.useGPU = false;
    fprintf('使用CPU计算\n');
end

%% 加载PSF数据
try
    a = load('I_exc532_501.mat');
    I1 = a.result.PSF;
    b = load('I_hexc488_501.mat');
    I2 = b.result.PSF;
    
    [LL, MM] = size(I1);
    
    % 归一化
    I1 = I1 / max(I1(:));
    I2 = I2 / max(I2(:));
    
    % 缩放光强
    I_exc = params.I_s * I1;
    I_hexc = params.I_d * I2;
    
catch ME
    error('无法加载PSF文件: %s', ME.message);
end

%% 仅提取中心行数据
center_row = ceil(LL/2);  % 第251行
fprintf('仅计算中心行数据 (第%d行)...\n', center_row);

% 提取中心行数据
I_exc_center = I_exc(center_row, :);
I_hexc_center = I_hexc(center_row, :);
num_points = length(I_exc_center);

%% 向量化计算中心行数据
fprintf('开始计算中心行数据 (%d个点)...\n', num_points);

% 将数据转移到GPU（如果可用）
if params.useGPU
    I_exc_center_gpu = gpuArray(I_exc_center);
    I_hexc_center_gpu = gpuArray(I_hexc_center);
    t_gpu = gpuArray(params.t);
else
    I_exc_center_gpu = I_exc_center;
    I_hexc_center_gpu = I_hexc_center;
    t_gpu = params.t;
end

% 预计算余弦信号
cos_f1 = cos(2*pi*params.f1*t_gpu);
cos_f2 = cos(2*pi*params.f2*t_gpu);

% 计算速率常数向量
k_s_vector = params.const_s * I_exc_center_gpu;
k_d_vector = params.const_d * I_hexc_center_gpu;

% 传统SAC计算（向量化）
y_SAC = k_s_vector ./ (params.c1*k_s_vector + params.k0 + params.c1*k_d_vector);

% 重塑为列向量以便向量化计算
k_s_col = reshape(k_s_vector, [], 1);
k_d_col = reshape(k_d_vector, [], 1);

% 扩展维度以便向量化计算
k_s_expanded = reshape(k_s_col, [num_points, 1]);
k_d_expanded = reshape(k_d_col, [num_points, 1]);

% 预计算扩展的余弦信号
cos_f1_expanded = reshape(cos_f1, [1, params.n_time]);
cos_f2_expanded = reshape(cos_f2, [1, params.n_time]);

fprintf('计算调制信号...\n');

% 向量化计算y_s_1、y_s_2和y_s_3
numerator = k_s_expanded .* (1 + params.m_s_1 * cos_f1_expanded);
denominator_1= params.c1 * (k_s_expanded .* (1 + params.m_s_1 * cos_f1_expanded) + ...
                   k_d_expanded .* (1 + params.m_d * cos_f2_expanded)) + params.k0;
y_s_1 = numerator ./ denominator_1;

numerator = k_s_expanded .* (1 + params.m_s_2 * cos_f1_expanded);
denominator_2= params.c1 * (k_s_expanded .* (1 + params.m_s_2 * cos_f1_expanded) + ...
                   k_d_expanded .* (1 + params.m_d * cos_f2_expanded)) + params.k0;
y_s_2 = numerator ./ denominator_2;

numerator = k_s_expanded .* (1 + params.m_s_3 * cos_f1_expanded);
denominator_3 = params.c1 * (k_s_expanded .* (1 + params.m_s_3 * cos_f1_expanded) + ...
                        k_d_expanded .* (1 + params.m_d * cos_f2_expanded)) + params.k0;
y_s_3 = numerator ./ denominator_3;

numerator = k_s_expanded .* (1 + params.m_s_4 * cos_f1_expanded);
denominator_4 = params.c1 * (k_s_expanded .* (1 + params.m_s_4 * cos_f1_expanded) + ...
                        k_d_expanded .* (1 + params.m_d * cos_f2_expanded)) + params.k0;
y_s_4 = numerator ./ denominator_4;

fprintf('进行FFT分析...\n');

% 批量FFT计算
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
    % CPU上的向量化FFT
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
    f_fft_shift_4 = fftshift(f_fft_4, 2);
    result_4 = abs(f_fft_shift_4) ./ max(abs(f_fft_shift_4), [], 2);
end

% 计算频率分量占比（向量化）
sumx_1 = (sum(result_1, 2) - result_1(:, params.center_freq_idx)) / 2;
sumx_2 = (sum(result_2, 2) - result_2(:, params.center_freq_idx)) / 2;
sumx_3 = (sum(result_3, 2) - result_3(:, params.center_freq_idx)) / 2;
sumx_4 = (sum(result_4, 2) - result_4(:, params.center_freq_idx)) / 2;

% 提取频率分量
sig_fund_1 = result_1(:, params.f1_idx) ./ sumx_1;
sig_fund_2 = result_2(:, params.f1_idx) ./ sumx_2;
sig_fund_3 = result_3(:, params.f1_idx) ./ sumx_3;
sig_fund_4 = result_4(:, params.f1_idx) ./ sumx_4;

sig_harm_1 = result_1(:, params.f2_idx) ./ sumx_1;
sig_harm_2 = result_2(:, params.f2_idx) ./ sumx_2;
sig_harm_3 = result_3(:, params.f2_idx) ./ sumx_3;
sig_harm_4 = result_4(:, params.f2_idx) ./ sumx_4;

%% 将数据移回CPU（如果使用了GPU）
if params.useGPU
    fprintf('将数据从GPU传输回CPU...\n');
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

%% 计算fmSAC_1、fmSAC_2和fmSAC_3
fprintf('计算fmSAC_1、fmSAC_2和fmSAC_3...\n');

% alpha_matrix = sig_fund_1 ./ sig_harm_1;
% beta_matrix = sig_fund_2 ./ sig_harm_2;
% gamma_matrix = sig_fund_3 ./ sig_harm_3;
% 
% alpha_val = min(alpha_matrix);
% beta_val = min(beta_matrix);
% gamma_val = min(gamma_matrix);
% 
% fmSAC_1 = sig_fund_1 - alpha_val * sig_harm_1;
% fmSAC_2 = sig_fund_2 - beta_val * sig_harm_2;
% fmSAC_3 = sig_fund_3 - gamma_val * sig_harm_3;

% 以调制系数m_d=1.0时的差分差分系数作为所有的调制系数
alpha_matrix = sig_fund_4 ./ sig_harm_4;
alpha_val = min(alpha_matrix);

fmSAC_1 = sig_fund_1 - alpha_val * sig_harm_1;
fmSAC_2 = sig_fund_2 - alpha_val * sig_harm_2;
fmSAC_3 = sig_fund_3 - alpha_val * sig_harm_3;

% 将fmSAC_1/fmSAC_2/fmSAC_3底部对齐
fmSAC_1 = fmSAC_1-min(fmSAC_1);
fmSAC_2 = fmSAC_2-min(fmSAC_2);
fmSAC_3 = fmSAC_3-min(fmSAC_3);

% 归一化传统SAC
y_SAC = y_SAC / max(y_SAC);

%% 去除旁瓣（仅对中心行）
% 创建位置向量
x_pos = 1:num_points;
center_pos = ceil(num_points/2);
radius = abs(x_pos - center_pos);

min_val_1 = min(fmSAC_1);
min_val_2 = min(fmSAC_2);
min_val_3 = min(fmSAC_3);
% 找到所有最小值点的位置
threshold_1 = find(fmSAC_1 == min_val_1);
threshold_2 = find(fmSAC_2 == min_val_2);
threshold_3 = find(fmSAC_3 == min_val_3);

% 使用相对阈值去除旁瓣
%系数0.27为根据501×501情况下的选择参数，需要根据尺寸有所调整
fmSAC_1(radius > abs(threshold_1 - center_pos)) = min(fmSAC_1);
fmSAC_2(radius > abs(threshold_2 - center_pos)) = min(fmSAC_2);
fmSAC_3(radius > abs(threshold_3 - center_pos)) = min(fmSAC_3);

% 求FWHM
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

%% 绘图 - 直接使用向量数据
fprintf('生成图表...\n');

% Figure 1: fmSAC_1
figure(1);
set(gcf, 'Position', [100, 100, 1400, 500]);  % 图窗设置（像素）
plot(sig_fund_1, 'linewidth', 2, 'DisplayName', 'Fund Freq_1(f_{1})');
hold on;
plot(sig_harm_1, 'linewidth', 2, 'DisplayName', 'Harm Freq_1(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_1, 'linewidth', 2, 'DisplayName', 'fmSAC_1');
hold off;
box on;
% 图形美化
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_1 (CM_1 = %.1f)', params.m_s_1),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 2: fmSAC_2
figure(2);
set(gcf, 'Position', [100, 100, 1400, 500]);  % 图窗设置（像素）
plot(sig_fund_2, 'linewidth', 2, 'DisplayName', 'Fund Freq_2(f_{1})');
hold on;
plot(sig_harm_3, 'linewidth', 2, 'DisplayName', 'Harm Freq_2(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_2, 'linewidth', 2, 'DisplayName', 'fmSAC_2');
hold off;
box on;
% 图形美化
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_2 (CM_1 = %.1f)', params.m_s_2),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 3: fmSAC_3
figure(3);
set(gcf, 'Position', [100, 100, 1400, 500]);  % 图窗设置（像素）
plot(sig_fund_3, 'linewidth', 2, 'DisplayName', 'Fund Freq_3(f_{1})');
hold on;
plot(sig_harm_3, 'linewidth', 2, 'DisplayName', 'Harm Freq_3(f_{2})');
plot(y_SAC, 'linewidth', 2, 'DisplayName', 'Conventional SAC');
plot(fmSAC_3, 'linewidth', 2, 'DisplayName', 'fmSAC_3');
hold off;
box on;
% 图形美化
xlim([0, 500]);
set(gca,'LineWidth',2,'FontWeight','bold','FontSize',18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Position (nm)','FontWeight','bold','FontSize',24);
title(sprintf('fmSAC_3 (CM_1 = %.1f)', params.m_s_3),'FontWeight', 'bold', 'FontSize', 24);
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;

% Figure 4: 对比
figure(4);
set(gcf, 'Position', [100, 100, 1400, 750]);  % 图窗设置（像素）
plot(y_SAC,'Color',[0.85,0.33,0.1],'linewidth',2,'DisplayName','Conventional SAC');
hold on;
plot(fmSAC_1,'Color',[0.47,0.67,0.19],'linewidth',2,'DisplayName',sprintf('fmSAC_1 (CM_1=%.1f)',params.m_s_1));
plot(fmSAC_2,'Color',[0.12,0.47,0.71],'linewidth',2,'DisplayName',sprintf('fmSAC_2 (CM_1=%.1f)',params.m_s_2));
plot(fmSAC_3,'Color',[0.58,0.4,0.74],'linewidth',2,'DisplayName',sprintf('fmSAC_3 (CM_1=%.1f)',params.m_s_3));
% 计算并绘制半高全宽横线
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
% 图形美化
xlim([0, 500]);
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Normalized Intensity (a.u.)', 'FontWeight', 'bold', 'FontSize', 24);
xlabel('Position (nm)','FontWeight', 'bold', 'FontSize', 24);
% title('调制对比度对fmSAC的影响');
legend('show', 'Box', 'off', 'FontWeight', 'bold', 'FontSize', 12,'Location', 'northeast');
grid on;


% 显示参数摘要
fprintf('\n=== 参数摘要 ===\n');
fprintf('物理参数:\n');
fprintf('  激发波长: %.1f nm, 竞争波长: %.1f nm\n', params.lambda_s*1e7, params.lambda_d*1e7);
fprintf('  激发光强: %.1f kW/cm², 竞争光强: %.1f kW/cm²\n', params.I_s/1e3, params.I_d/1e3);
fprintf('调制参数:\n');
fprintf('  竞争调制对比度 m_d: %.1f\n', params.m_d);
fprintf('  激发1调制对比度 m_s: %.1f (fmSAC_1)\n', params.m_s_1);
fprintf('  激发2调制对比度 m_s: %.1f (fmSAC_2)\n', params.m_s_2);
fprintf('  激发3调制对比度 m_s: %.1f (fmSAC_3)\n', params.m_s_3);
fprintf('  竞争1半高全宽 FWHM: %.1f nm (FWHM_1)\n', FWHM_1);
fprintf('  竞争2半高全宽 FWHM: %.1f nm (FWHM_2)\n', FWHM_2);
fprintf('  竞争3半高全宽 FWHM: %.1f nm (FWHM_3)\n', FWHM_3);
fprintf('  传统fmSAC半高全宽 FWHM: %.1f nm (FWHM)\n', FWHM);
fprintf('频率参数:\n');
fprintf('  f1: %.1f kHz, f2: %.1f kHz\n', params.f1/1e3, params.f2/1e3);
fprintf('计算优化:\n');
fprintf('  计算点数从 %d 减少到 %d，加速比约为 %.1f倍\n', ...
        LL*MM, num_points, (LL*MM)/num_points);
fprintf('分析完成！\n');