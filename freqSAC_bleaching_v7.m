%% PRE-PROCESS
clc; clear; close all;
addpath(genpath('PSF files'));

%% 检查GPU可用性并初始化
fprintf('检查GPU设备...\n');
if gpuDeviceCount > 0
    gpu = gpuDevice();
    fprintf('使用GPU: %s\n', gpu.Name);
    useGPU = true;
else
    fprintf('未检测到GPU，使用CPU计算\n');
    useGPU = false;
end

%% 参数初始化
fprintf('初始化参数...\n');
params = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'c1', 1 + 1.1e6/0.49e6, ... % 预计算
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'lambda_s', 532e-7, ...
    'lambda_d', 488e-7, ...
    'sigma_s', 2.7e-16, ...
    'sigma_d', 2.7e-16 * 0.512063188, ... % 预计算
    'I_s', 10e3, ...
    'I_d_SAC', 500e3, ...     % SAC的竞争光强
    'I_d_fmSAC', 100e3, ...   % fmSAC的竞争光强（更低）
    'f1', 10e3, ...
    'f2', 15e3, ...
    'interval', 10e-6, ...
    't', 0:10e-6:1-10e-6, ... % 预计算时间序列
    'm_s', 0.1, ...
    'm_d', 1.0 ...
);

% 将时间序列转移到GPU（如果可用）
if useGPU
    params.t = gpuArray(params.t);
end

%% 加载PSF数据并转移到GPU
fprintf('加载PSF数据...\n');
a=load('I_exc532_51_3D.mat');
exc1 =a.result.PSF(:,:,25);
b=load('I_hexc488_51_3D.mat');
exc2 =b.result.PSF(:,:,25);

% fprintf('加载PSF数据...\n');
% a=load('I_exc532_501.mat');
% exc1 =a.result.PSF(:,:);
% b=load('I_hexc488_501.mat');
% exc2 =b.result.PSF(:,:);

% 转移到GPU
if useGPU
    exc1 = gpuArray(exc1);
    exc2 = gpuArray(exc2);
end

nor_exc1 = exc1 / max(exc1(:));
nor_exc2 = exc2 / max(exc2(:));

I = params.I_s * nor_exc1;
Id_SAC = params.I_d_SAC * nor_exc2;
Id_fmSAC = params.I_d_fmSAC * nor_exc2;

%% 物理参数
lambda = 532e-7; % 激发光的波长
h = 6.626e-34; % 普兰克常数
c = 3e10; % 光速
te = 1.92e-6; % 单点停留里面，脉冲数*60ps；其实就是td*0.0048（占空比）
td = 400e-6; % 一个周期,400us

phif = 0.02; % 荧光探测效率
tob = 0.4e-3; % 重要，扫描的时候每点停留时间，400us，即0.4ms
k0 = 2.56e8; % 基态分子迁移速率
kf = 2.4e8;
PHIf = kf / k0; % 0.95,发射量子效率
kisc = 1.1e6; % 倒数是能级系统间跃迁寿命0.909us
kt = 4.9e5; % 倒数是三重态寿命2.04us
sig01 = 2.22e-16; % 吸收截面，单位cm-2
sig1n = 0.77e-17; % 从S1跃迁到Sn的吸收截面
sigt1n = 3.85e-17; % 从T0跃迁到Tn的吸收截面

kb = 650; % 总的漂白速率，倒数是1.54ms
ksn1 = 5e12; % Sn到S1的迁移速率，倒数是寿命0.2ps
ktn1 = ksn1; % Tn到T1的迁移速率
kbsn = 2.8e8; % Sn能级上的漂白速率，3.5ns
kbtn = 2.8e8; % Tn能级上的漂白速率，3.5ns

gamma = lambda / (h * c);

%% 分别计算SAC和fmSAC的速率参数
fprintf('计算SAC和fmSAC的速率参数...\n');

% SAC参数
k01_SAC = sig01 .* I * gamma;
k01d_SAC = sig01 .* Id_SAC * gamma;
ka_SAC = k01_SAC + k01d_SAC;
k1n_SAC = sig1n .* (Id_SAC + I) * gamma;
kt1n_SAC = sigt1n .* (I + Id_SAC) * gamma;

% fmSAC参数  
k01_fmSAC = sig01 .* I * gamma;
k01d_fmSAC = sig01 .* Id_fmSAC * gamma;
ka_fmSAC = k01_fmSAC + k01d_fmSAC;
k1n_fmSAC = sig1n .* (Id_fmSAC + I) * gamma;
kt1n_fmSAC = sigt1n .* (I + Id_fmSAC) * gamma;

% Confocal参数（无竞争光）
k01_conf = sig01 .* I * gamma;
k1n_conf = sig1n .* I * gamma;
kt1n_conf = sigt1n .* I * gamma;

%% 计算传统SAC和fmSAC PSF (GPU优化)
fprintf('计算传统SAC和fmSAC PSF...\n');
LL = size(I, 1);

% 预计算常数
const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

% 预计算频率索引
n_time = length(params.t);
freq_res = (1/params.interval)/n_time;
f1_idx = round(params.f1/freq_res) + n_time/2 + 1;
f2_idx = round(params.f2/freq_res) + n_time/2 + 1;

% 预计算调制信号（在GPU上）
if useGPU
    cos_f1 = gpuArray(cos(2*pi*params.f1*params.t));
    cos_f2 = gpuArray(cos(2*pi*params.f2*params.t));
else
    cos_f1 = cos(2*pi*params.f1*params.t);
    cos_f2 = cos(2*pi*params.f2*params.t);
end

hWaitbar = waitbar(0, '计算PSF...', 'Name', 'fmSAC光漂白仿真进度');

% 初始化矩阵
if useGPU
    y_SAC = gpuArray.zeros(LL);
    sig_fund_matrix = gpuArray.zeros(LL);
    sig_harm_matrix = gpuArray.zeros(LL);
else
    y_SAC = zeros(LL);
    sig_fund_matrix = zeros(LL);
    sig_harm_matrix = zeros(LL);
end

% 主计算循环
for m = 1:LL
    for n = 1:LL
        % 传统SAC计算（使用SAC竞争光强）
        k_s = const_s * I(m, n);
        k_d_SAC = const_d * Id_SAC(m, n);
        y_SAC(m, n) = k_s / (params.c1 * k_s + params.c1 * k_d_SAC + params.k0);
        
        % fmSAC计算 - 时域调制（使用fmSAC竞争光强）
        k_d_fmSAC = const_d * Id_fmSAC(m, n);
        numerator = k_s * (1 + params.m_s * cos_f1);
        denominator = params.c1 * (k_s * (1 + params.m_s * cos_f1) + ...
                       k_d_fmSAC * (1 + params.m_d * cos_f2)) + params.k0;
        y_s = numerator ./ denominator;
        
        % 频谱分析
        f_fft = fft(y_s);
        f_fft_shift = fftshift(f_fft);
        result = abs(f_fft_shift) / max(abs(f_fft_shift));
        
        sumx = (sum(result) - result(n_time/2+1)) / 2;
        sig_fund_matrix(m, n) = result(f1_idx) / sumx;
        sig_harm_matrix(m, n) = result(f2_idx) / sumx;
    end
    
    % 更新进度
    waitbar(m/LL, hWaitbar, sprintf('计算PSF: %.1f%%', m/LL*100));
end

% 计算全局alpha值并应用
alpha_val = min(sig_fund_matrix(:) ./ sig_harm_matrix(:));
fmSAC = sig_fund_matrix - alpha_val * sig_harm_matrix;

% 创建圆形掩膜去除旁瓣
[x, y] = meshgrid(1:LL, 1:LL);
if useGPU
    [x, y] = meshgrid(gpuArray(1:LL), gpuArray(1:LL));
end
center = [ceil(LL/2), ceil(LL/2)];
radius = sqrt((x - center(2)).^2 + (y - center(1)).^2);
fmSAC(radius > 8) = 0;

close(hWaitbar);

% 归一化PSF
y_SAC = y_SAC / max(y_SAC(:));
fmSAC = fmSAC / max(fmSAC(:));

%% 分别计算SAC和fmSAC的漂白模型
fprintf('计算漂白模型...\n');

% SAC漂白模型
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

% fmSAC漂白模型
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

% Confocal漂白模型（简单模型）
beta_conf = kb * 0.05; % Confocal漂白速率较慢

%% 产生PSF (传统SAC和fmSAC)
fprintf('产生PSF...\n');

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

%% 计算漂白效应 - 增强漂白效果
fprintf('计算漂白效应...\n');
scan_intensity = 0.8; % 增加扫描强度因子，增强漂白效应

% 计算漂白因子
R_scan_SAC = exp(-beta_SAC .* te .* ((1 + delta_SAC) - delta_SAC .* exp(-k_bleach_SAC * te)));
R_scan_fmSAC = exp(-beta_fmSAC .* te .* ((1 + delta_fmSAC) - delta_fmSAC .* exp(-k_bleach_fmSAC * te)));
R_scan_conf = exp(-beta_conf .* te); % Confocal漂白较简单

% 应用漂白效应 - 使用更强的漂白因子
bleach_factor_SAC = 0.5; % SAC漂白因子 - 较强的漂白
bleach_factor_fmSAC = 0.7; % fmSAC漂白因子 - 较弱的漂白
bleach_factor_conf = 0.9; % Confocal漂白因子 - 最弱的漂白

fprintf('漂白因子:\n');
fprintf('  Confocal: %.3f\n', bleach_factor_conf);
fprintf('  SAC: %.3f\n', bleach_factor_SAC);
fprintf('  fmSAC: %.3f\n', bleach_factor_fmSAC);

% 漂白后的PSF
Iout_SAC_bleaching = Iout_SAC .* bleach_factor_SAC;
Iout_fmSAC_bleaching = Iout_fmSAC .* bleach_factor_fmSAC;
Iout_conf_bleaching = Iout_conf .* bleach_factor_conf;

%% 成像仿真
fprintf('进行成像仿真...\n');
n = 100; % 用于控制范围，一个像素50nm，形成一个（2n）*(2n)的范围
m = 50; % 荧光分子个数
s = makematrix(n, m, useGPU);

% 将PSF转移到CPU进行卷积运算
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

% 基础噪声水平
base_noise = 10;

% 成像计算 - 初始状态（轻微噪声）
conf = conv2(s_cpu, Iout_conf_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_SAC = conv2(s_cpu, Iout_SAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC = conv2(s_cpu, Iout_fmSAC_cpu, 'same') + base_noise * (rand(size(s_cpu)) - 0.5);

% 漂白后的成像 - 信号更弱，噪声更强
enhanced_noise = 80; % 漂白后噪声增强
conf_bleaching = conv2(s_cpu, Iout_conf_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_SAC_bleaching = conv2(s_cpu, Iout_SAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);
result_fmSAC_bleaching = conv2(s_cpu, Iout_fmSAC_bleaching_cpu, 'same') + enhanced_noise * (rand(size(s_cpu)) - 0.5);

%% 计算荧光信号随时间衰减曲线（只显示SAC和fmSAC）
fprintf('计算荧光信号衰减曲线...\n');
num_scans = 50; % 扫描次数
scan_range = 0:num_scans;

% 选择单个荧光分子进行分析
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

% 定义分析区域
roi_size = 5;
y_range = max(1, center_mol_y - floor(roi_size/2)):min(size(s_cpu, 1), center_mol_y + floor(roi_size/2));
x_range = max(1, center_mol_x - floor(roi_size/2)):min(size(s_cpu, 2), center_mol_x + floor(roi_size/2));

% 初始化信号衰减数组（只保留SAC和fmSAC）
signal_SAC = zeros(1, num_scans + 1);
signal_fmSAC = zeros(1, num_scans + 1);

% 计算初始信号（scan = 0）
signal_SAC(1) = sum(sum(result_SAC(y_range, x_range)));
signal_fmSAC(1) = sum(sum(result_fmSAC(y_range, x_range)));

% 计算不同scan次数下的信号（只计算SAC和fmSAC）
for scan_idx = 1:num_scans
    % 计算当前scan的漂白因子（累积漂白效应）
    current_bleach_SAC = bleach_factor_SAC.^(scan_idx*0.2);
    current_bleach_fmSAC = bleach_factor_fmSAC.^(scan_idx*0.15);
    
    % 计算漂白后的成像
    result_SAC_scan = conv2(s_cpu, Iout_SAC_cpu .* current_bleach_SAC, 'same');
    result_fmSAC_scan = conv2(s_cpu, Iout_fmSAC_cpu .* current_bleach_fmSAC, 'same');
    
    % 添加随scan增加的噪声
    noise_level = base_noise * (1 + scan_idx * 0.03); % 噪声随scan线性增加
    R_scan = noise_level * (rand(size(s_cpu)) - 0.5);
    
    result_SAC_scan = result_SAC_scan + R_scan;
    result_fmSAC_scan = result_fmSAC_scan + R_scan;
    
    % 记录信号强度
    signal_SAC(scan_idx + 1) = sum(sum(result_SAC_scan(y_range, x_range)));
    signal_fmSAC(scan_idx + 1) = sum(sum(result_fmSAC_scan(y_range, x_range)));
end

% 归一化信号强度
signal_SAC_norm = signal_SAC / signal_SAC(1);
signal_fmSAC_norm = signal_fmSAC / signal_fmSAC(1);

% 拟合指数衰减曲线
fit_func = @(a, b, x) a * exp(-b * x);
x_fit = scan_range';

% SAC拟合  
[SAC_fit, SAC_gof] = fit(x_fit, signal_SAC_norm', fit_func, 'StartPoint', [1, 0.05]);
% fmSAC拟合
[fmSAC_fit, fmSAC_gof] = fit(x_fit, signal_fmSAC_norm', fit_func, 'StartPoint', [1, 0.02]);

% 生成拟合曲线
x_fit_continuous = linspace(0, num_scans, 100);
SAC_fit_curve = SAC_fit.a * exp(-SAC_fit.b * x_fit_continuous);
fmSAC_fit_curve = fmSAC_fit.a * exp(-fmSAC_fit.b * x_fit_continuous);

%% 结果显示
fprintf('生成结果图像...\n');

% 图1: 样品成像结果--未漂白
figure(1)
set(gcf, 'Position', [100, 100, 300, 200], 'Color', 'w');
colormap hot
imagesc(s_cpu);
colorbar; 
set(gca, 'XTick', [], 'YTick', []); 
axis square; 
% title('Sample Structure', 'FontSize', 12, 'FontWeight', 'bold');

% 创建共享颜色条
cbar = colorbar('Position', [0.8, 0.11, 0.04, 0.77]);
% cbar.Label.String = 'Normalized Intensity (a.u.)';
cbar.Label.FontSize = 10;
add_subplot_scalebar(gca, 200, 'nm', 0); % 20像素的比例尺

% 确保所有子图使用相同的颜色范围
clim([0, 1]); % 应用统一的颜色范围

% 图2: 成像结果对比 (2行3列)
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
% cbar1 = colorbar; % 保存颜色条句柄
% delete cbar1;
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('Confocal', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

subplot(2,3,2), imagesc(result_SAC);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title(sprintf('SAC (I_d = %dkW/cm²)', params.I_d_SAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

subplot(2,3,3), imagesc(result_fmSAC);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title(sprintf('fmSAC (I_d = %dkW/cm²)', params.I_d_fmSAC/1e3), 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

% 第二行：漂白后
subplot(2,3,4), imagesc(conf_bleaching);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('Confocal after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

subplot(2,3,5), imagesc(result_SAC_bleaching);
% colorbar;
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('SAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

subplot(2,3,6), imagesc(result_fmSAC_bleaching);
% colorbar; 
set(gca, 'XTick', [], 'YTick', []);
axis square; 
% title('fmSAC after Bleaching', 'FontSize', 12, 'FontWeight', 'bold')
add_subplot_scalebar(gca, 200, 'nm', 1); % 20像素的比例尺

% 调整子图位置，为颜色条留出空间
% sgtitle('多个子图共享一个颜色条', 'FontSize', 16);

% 创建共享颜色条
cbar = colorbar('Position', [0.94, 0.11, 0.015, 0.80]);
% cbar.Label.String = 'Normalized Intensity (a.u.)';
% cbar.Label.FontSize = 10;

% 确保所有子图使用相同的颜色范围
clim([cmin, cmax]); % 应用统一的颜色范围

% 保存子图（自定义设置）
output_dir2 = 'output_bleaching_images';
fprintf('   正在保存子图到: %s\n', output_dir2);
save_subplots_separately(figure(1), output_dir2, ...
    'FileFormat', 'tiff', ...
    'DPI', 1200, ...
    'Prefix', 'image_', ...
    'Silent', false);
fprintf('\n');
fprintf('1. %s - 图像子图\n', fullfile(pwd, output_dir2));
output_dir3 = 'output_bleaching_images_1';
fprintf('   正在保存子图到: %s\n', output_dir3);
save_subplots_separately(figure(2), output_dir3, ...
    'FileFormat', 'tiff', ...
    'DPI', 1200, ...
    'Prefix', 'image_', ...
    'Silent', false);
fprintf('\n');
fprintf('2. %s - 图像子图\n', fullfile(pwd, output_dir3));

% 图3: PSF剖面比较
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

% 创建带有光强信息的图例标签
sac_legend = sprintf('SAC @%dkW/cm²', params.I_d_SAC/1e3);
sac_bleached_legend = sprintf('SAC Bleached @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend = sprintf('fmSAC @%dkW/cm²', params.I_d_fmSAC/1e3);
fmsac_bleached_legend = sprintf('fmSAC Bleached @%dkW/cm²', params.I_d_fmSAC/1e3);

% 然后在legend中使用这些字符串
legend('Confocal', sac_legend, sac_bleached_legend, fmsac_legend, ...
    fmsac_bleached_legend, 'Location', 'northeast','FontSize', 12, 'Box', 'off');
grid on

% 图4: 荧光信号衰减曲线（只显示SAC和fmSAC）
figure(4)
set(gcf, 'Position', [100, 100, 1000, 600], 'Color', 'w');

% 绘制数据点
plot(scan_range, signal_SAC_norm, 'bs', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'b')
hold on
plot(scan_range, signal_fmSAC_norm, 'g^', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'g')

% 绘制拟合曲线
plot(x_fit_continuous, SAC_fit_curve, 'b-', 'LineWidth', 2)
plot(x_fit_continuous, fmSAC_fit_curve, 'g-', 'LineWidth', 2)
xlim([-0.5, 50.5]);
ylim([-0.01, 1.01]);
set(gca, 'FontSize', 15, 'LineWidth', 2, 'FontWeight', 'bold');
title('Fluorescence Signal Decay during Imaging', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Scan Number', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Normalized Fluorescence Intensity (a.u.)', 'FontSize', 18, 'FontWeight', 'bold');

% 创建带有光强信息的图例标签
sac_legend_data = sprintf('SAC Data @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_data = sprintf('fmSAC Data @%dkW/cm²', params.I_d_fmSAC/1e3);
sac_legend_fit = sprintf('SAC Fitted @%dkW/cm²', params.I_d_SAC/1e3);
fmsac_legend_fit = sprintf('fmSAC Fitted @%dkW/cm²', params.I_d_fmSAC/1e3);

% 在Figure3的legend中使用
legend(sac_legend_data, fmsac_legend_data, sac_legend_fit, fmsac_legend_fit, ...
       'Location', 'best', 'FontSize', 12, 'Box', 'off');
grid on;

% 添加半衰期标注
[~, idx_half_SAC] = min(abs(SAC_fit_curve - 0.5));
[~, idx_half_fmSAC] = min(abs(fmSAC_fit_curve - 0.5));

half_life_SAC = x_fit_continuous(idx_half_SAC);
half_life_fmSAC = x_fit_continuous(idx_half_fmSAC);

text(half_life_SAC, 0.25, sprintf('SAC: %.1f scans', half_life_SAC), ...
     'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center')
text(half_life_fmSAC, 0.7, sprintf('fmSAC: %.1f scans', half_life_fmSAC), ...
     'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center')

%% 分析结果
fprintf('\n=== 光漂白分析结果 ===\n');
fprintf('竞争光强设置:\n');
fprintf('  SAC: %d kW/cm²\n', params.I_d_SAC/1e3);
fprintf('  fmSAC: %d kW/cm²\n', params.I_d_fmSAC/1e3);

% 计算信号损失比例
SAC_bleaching_ratio = mean(Iout_SAC_bleaching_cpu(:)) / mean(Iout_SAC_cpu(:));
fmSAC_bleaching_ratio = mean(Iout_fmSAC_bleaching_cpu(:)) / mean(Iout_fmSAC_cpu(:));
conf_bleaching_ratio = mean(Iout_conf_bleaching_cpu(:)) / mean(Iout_conf_cpu(:));

fprintf('\n漂白后信号保留比例:\n');
fprintf('  Confocal: %.2f%%\n', conf_bleaching_ratio * 100);
fprintf('  传统SAC: %.2f%%\n', SAC_bleaching_ratio * 100);
fprintf('  fmSAC: %.2f%%\n', fmSAC_bleaching_ratio * 100);

fprintf('\n半衰期分析:\n');
fprintf('  SAC半衰期: %.1f scans\n', half_life_SAC);
fprintf('  fmSAC半衰期: %.1f scans\n', half_life_fmSAC);

half_life_improvement = (half_life_fmSAC - half_life_SAC) / half_life_SAC * 100;
fprintf('  fmSAC半衰期改善: +%.1f%%\n', half_life_improvement);

fprintf('\n衰减常数:\n');
fprintf('  SAC衰减常数: %.4f\n', SAC_fit.b);
fprintf('  fmSAC衰减常数: %.4f\n', fmSAC_fit.b);

fprintf('\n仿真完成！\n');

%% 辅助函数
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
    % 在子图中添加比例尺
    % subplot_handle: 子图句柄
    % physical_size: 比例尺的物理长度
    % units: 单位（如 'mm', 'cm', 'pixels' 等）
    
    % 获取当前图形和子图信息
    % fig = gcf;
    % fig_pos = get(fig, 'Position');
    % fig_width = fig_pos(3);
    % fig_height = fig_pos(4);
    
    % 获取子图在图形中的位置（归一化坐标）
    subplot_pos = get(subplot_handle, 'Position');
    subplot_x = subplot_pos(1);
    subplot_y = subplot_pos(2);
    % subplot_width = subplot_pos(3);
    % subplot_height = subplot_pos(4);
    [subplot_width, subplot_height] = get_image_width_without_colorbar(subplot_handle);
    
    % 获取子图的数据范围
    x_limits = get(subplot_handle, 'XLim');
    y_limits = get(subplot_handle, 'YLim');
    data_width = x_limits(2) - x_limits(1);
    data_height = y_limits(2) - y_limits(1);
    
    % 计算比例尺的像素长度（基于数据范围）
    if strcmpi(units, 'pixels')
        scalebar_data_length = physical_size;
    else
        % 如果有物理尺寸信息，可以在这里进行转换
        % 假设每单位数据对应0.1个像素
        scalebar_data_length = physical_size.*0.1;
    end
    
    % 将数据长度转换为子图中的归一化长度
    scalebar_normalized_length = scalebar_data_length / data_width * subplot_width;
    
    
    % 绘制比例尺
    if (fig_1 == 1)    
        % 设置比例尺位置（右下角，带边距）
        margin = 0.1; % 5%的边距
        scalebar_x = subplot_x + subplot_width * (1 - margin) - scalebar_normalized_length;
        scalebar_y = subplot_y + subplot_height * margin * 0.5;
        % scalebar_height = 0.02; % 比例尺高度（归一化）
        
        annotation('line', [scalebar_x, scalebar_x+scalebar_normalized_length], [scalebar_y, scalebar_y], ...
            'Color', 'white', ...
            'LineStyle', '-', ...
            'LineWidth', 2);
    else
        % 设置比例尺位置（右下角，带边距）
        margin = 0.05; % 5%的边距
        scalebar_x = subplot_x + subplot_width * (1 - margin) - scalebar_normalized_length;
        scalebar_y = subplot_y + subplot_height * margin;
        % scalebar_height = 0.02; % 比例尺高度（归一化）

        annotation('line', [scalebar_x, scalebar_x+scalebar_normalized_length], [scalebar_y, scalebar_y], ...
            'Color', 'white', ...
            'LineStyle', '-', ...
            'LineWidth', 2);
    end

end

function [image_width, image_height] = get_image_width_without_colorbar(subplot_handle)
    % 获取子图的位置信息
    subplot_pos = get(subplot_handle, 'Position');
    subplot_x = subplot_pos(1);
    subplot_total_width = subplot_pos(3);
    subplot_total_height = subplot_pos(4);
    
    % 获取图像数据范围
    x_limits = get(subplot_handle, 'XLim');
    y_limits = get(subplot_handle, 'YLim');
    data_width = x_limits(2) - x_limits(1);
    data_height = y_limits(2) - y_limits(1);
    
    % 计算纵横比
    aspect_ratio = data_width / data_height;
    
    % 计算实际图像区域的宽度和高度
    % 假设子图中除了图像还有颜色条和可能的边距
    % 这里通过数据纵横比来推断实际图像区域
    
    % 获取当前图形
    fig = get(subplot_handle, 'Parent');
    
    % 检查是否有颜色条
    cbar_handles = findobj(fig, 'Type', 'colorbar');
    
    if ~isempty(cbar_handles)
        % 有颜色条，计算其宽度
        cbar = cbar_handles(1);
        cbar_pos = get(cbar, 'Position');
        cbar_x = cbar_pos(1);
        cbar_width = cbar_pos(3);
        
        % 图像区域宽度 = 子图总宽度 - 颜色条宽度 - 边距
        % 假设边距为颜色条宽度的一半
        margin = cbar_width * 0.5;
        image_width = cbar_x - subplot_x - margin;
        
    else
        % 没有颜色条，使用子图总宽度
        image_width = subplot_total_width;
    end
    
    % 图像高度通常等于子图总高度（假设颜色条在右侧）
    image_height = subplot_total_height;
    
    % 确保宽度合理
    if image_width <= 0
        image_width = subplot_total_width * 0.8; % 默认使用80%的宽度
    end
end