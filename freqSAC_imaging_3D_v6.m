%% fmSAC PSF仿真/分子成像/微管成像 - 3D版本（支持切片计算）
% 包含Confocal、SAC、fmSAC的3D PSF和成像仿真
% 设置use_3d为false，进行3D切片，slice_z可设置为1~51；在"% 分子成像仿真"部分，选择'random_2d'或'microtubule_2d'
% 设置use_3d为true，进行3D切片；在"分子成像仿真"部分，选择'random_3d'或'microtubule_3d'
clc; clear; close all;
addpath(genpath('PSF'));
tic;
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
    'I_d', 500e3, ...
    'f1', 10e3, ...
    'f2', 15e3, ...
    'interval', 10e-6, ...
    't', 0:10e-6:1-10e-6, ... % 预计算时间序列
    'm_s', 0.1, ...
    'm_d', 1.0, ...
    'use_3d', true, ... % 设置为true进行3D计算，false进行2D切片计算
    'slice_z', 1, ... % 当use_3d为false时，使用的Z轴切片索引;slice_z为25或26，为焦面的切片
    'SidelobeCoeff',8 ...% 用于控制fmSAC去除旁瓣的参数，对于51×51×51的PSF，推荐用8
);

%% 加载3D PSF数据
fprintf('加载3D PSF数据...\n');
try
    % 请替换为您的实际3D PSF数据文件
    exc_data = load('I_exc532_51_3D.mat'); % 替换为您的激发光PSF文件
    
    hexc_data = load('I_hexc488_51_3D.mat'); % 替换为您的竞争光PSF文件
  
    % 提取PSF数据，根据您的数据结构调整字段名
    I_exc_psf = exc_data.result.PSF; % 如果字段名不同，请调整
    I_hexc_psf = hexc_data.result.PSF; % 如果字段名不同，请调整
    
    fprintf('PSF数据加载成功\n');
    fprintf('激发光PSF尺寸: %s\n', mat2str(size(I_exc_psf)));
    fprintf('竞争光PSF尺寸: %s\n', mat2str(size(I_hexc_psf)));
    
catch ME
    fprintf('加载PSF数据失败: %s\n', ME.message);
    fprintf('创建示例3D PSF数据...\n');
    
    % 创建示例3D PSF数据（高斯分布）
    [X, Y, Z] = meshgrid(-25:25, -25:25, -25:25);
   
    I_exc_psf = exp(-(X.^2 + Y.^2 + Z.^2)/(2*10^2));% Gauss光斑
    theta = atan2(Y, X); % 角向坐标
    I_hexc_psf = exp(-(X.^2 + Y.^2)/(2*8^2)) .* (X.^2 + Y.^2) .* exp(1i*theta) .* exp(-Z.^2/(2*6^2)); % 带拓扑荷的Vortex光斑
    I_hexc_psf = abs(I_hexc_psf); % 取强度
end

%{
% 检查是否需要切片计算
if ~params.use_3d
    fprintf('使用2D切片计算，Z轴切片索引: %d\n', params.slice_z);
    
    % 提取指定切片
    if size(I_exc_psf, 3) > 1
        I_exc_psf = I_exc_psf(:, :, params.slice_z);
        I_hexc_psf = I_hexc_psf(:, :, params.slice_z);
    end
end
%}

% 归一化并缩放
I1 = I_exc_psf / max(I_exc_psf(:));
I2 = I_hexc_psf / max(I_hexc_psf(:));
I_exc = params.I_s * I1;
I_hexc = params.I_d * I2;

% 获取数据尺寸
% if params.use_3d
[LL, MM, NN] = size(I_exc);
fprintf('3D PSF尺寸: %d x %d x %d\n', LL, MM, NN);
%{
else
    [LL, MM] = size(I_exc(:, :, params.slice_z));
    fprintf('2D切片PSF尺寸: %d x %d\n', LL, MM);
end
%}

%% 预计算常数（提高效率）
fprintf('预计算常数...\n');
const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

%% 初始化进度条
fprintf('开始计算传统SAC PSF...\n');
hWaitbar = waitbar(0, '计算传统SAC PSF...', 'Name', 'fmSAC仿真进度');

%% 计算传统SAC PSF
% if params.use_3d
% 3D计算 - GPU加速
I_exc_gpu = gpuArray(I_exc);
I_hexc_gpu = gpuArray(I_hexc);
    
k_s = const_s * I_exc_gpu;
k_d = const_d * I_hexc_gpu;
y_SAC_gpu = k_s ./ (params.c1 * k_s + params.c1 * k_d + params.k0);
y_SAC = gather(y_SAC_gpu);
    
y_SAC = y_SAC / max(y_SAC(:));
    
waitbar(1, hWaitbar, '传统SAC PSF计算完成');
%{
else 
    % 2D切片计算
    y_SAC = zeros(LL, MM);
    total_pixels = LL * MM;
    
    for m = 1:LL
        for n = 1:MM
            k_s = const_s * I_exc(m, n);
            k_d = const_d * I_hexc(m, n);
            y_SAC(m, n) = k_s / (params.c1 * k_s + params.c1 * k_d + params.k0);
            
            % 更新进度
            if mod((m-1)*MM + n, 100) == 0
                progress = ((m-1)*MM + n) / total_pixels;
                waitbar(progress, hWaitbar, sprintf('计算传统SAC PSF: %.1f%%', progress*100));
            end
        end
    end
    
    y_SAC = y_SAC / max(y_SAC(:));
end
%}

%% 计算fmSAC
fprintf('\n计算fmSAC PSF...\n');

% 预计算频率索引（提高效率）
n_time = length(params.t);
freq_res = (1/params.interval)/n_time;
f1_idx = round(params.f1/freq_res) + n_time/2 + 1;
f2_idx = round(params.f2/freq_res) + n_time/2 + 1;

waitbar(0.5, hWaitbar, sprintf('计算fmSAC @ %dkW/cm²...', params.I_d/1e3));

% if params.use_3d
% 3D计算 - GPU加速
I_exc_gpu = gpuArray(I_exc);
I_hexc_gpu = gpuArray(I_hexc);
    
% 预计算调制信号
cos_f1 = cos(2*pi*params.f1*params.t);
cos_f2 = cos(2*pi*params.f2*params.t);
    
% 初始化GPU数组
sig_fund_gpu = gpuArray.zeros(LL, MM, NN);
sig_harm_gpu = gpuArray.zeros(LL, MM, NN);
    
% 逐点计算
for m = 1:LL
    for n = 1:MM
        for p = 1:NN
            k_s = const_s * I_exc_gpu(m, n, p);
            k_d = const_d * I_hexc_gpu(m, n, p);
                
            % 生成调制信号
            numerator = k_s * (1 + params.m_s * cos_f1);
            denominator = params.c1*(k_s*(1 + params.m_s * cos_f1) + ...
                k_d*(1 + params.m_d * cos_f2)) + params.k0;
            y_s = numerator ./ denominator;
                    
            % 频谱分析
            f_fft = fft(y_s);
            f_fft_shift = fftshift(f_fft);
            result = abs(f_fft_shift) / max(abs(f_fft_shift));
                    
            sumx = (sum(result) - result(n_time/2+1)) / 2;
            sig_fund_gpu(m, n, p) = result(f1_idx) / sumx;
            sig_harm_gpu(m, n, p) = result(f2_idx) / sumx;
        end
    end
        
    % 更新进度
    progress = 0.5 + (m/LL)*0.4;
    waitbar(progress, hWaitbar, sprintf('计算fmSAC: %.1f%%', m/LL*100));
end
    
% 回传数据到CPU
sig_fund = gather(sig_fund_gpu);
sig_harm = gather(sig_harm_gpu);
    
% 计算fmSAC并去除旁瓣
alpha_val = min(sig_fund(:) ./ sig_harm(:));
fmSAC = sig_fund - alpha_val * sig_harm;
    
% 创建掩膜去除旁瓣
[x, y, z] = meshgrid(1:MM, 1:LL, 1:NN);
center = [ceil(LL/2), ceil(MM/2), ceil(NN/2)];
radius = sqrt((x - center(2)).^2 + (y - center(1)).^2 + (z - center(3)).^2);
fmSAC(radius > params.SidelobeCoeff) = 0;
    
%{
 else
    % 2D切片计算
    sig_fund = zeros(LL, MM);
    sig_harm = zeros(LL, MM);
    
    for m = 1:LL
        for n = 1:MM
            k_s = const_s * I_exc(m, n);
            k_d = const_d * I_hexc(m, n);
            
            % 生成调制信号
            cos_f1 = cos(2*pi*params.f1*params.t);
            cos_f2 = cos(2*pi*params.f2*params.t);
            
            numerator = k_s * (1 + params.m_s * cos_f1);
            denominator = params.c1*(k_s*(1 + params.m_s * cos_f1) + ...
                           k_d*(1 + params.m_d * cos_f2)) + params.k0;
            y_s = numerator ./ denominator;
            
            % 频谱分析
            f_fft = fft(y_s);
            f_fft_shift = fftshift(f_fft);
            result = abs(f_fft_shift) / max(abs(f_fft_shift));
            
            sumx = (sum(result) - result(n_time/2+1)) / 2;
            sig_fund(m, n) = result(f1_idx) / sumx;
            sig_harm(m, n) = result(f2_idx) / sumx;
        end
        
        % 更新进度
        progress = 0.5 + (m/LL)*0.4;
        waitbar(progress, hWaitbar, sprintf('计算fmSAC: %.1f%%', m/LL*100));
    end
    
    % 计算fmSAC并去除旁瓣
    alpha_val = min(sig_fund(:) ./ sig_harm(:));
    fmSAC = sig_fund - alpha_val * sig_harm;
    
    % 创建圆形掩膜去除旁瓣
    [x, y] = meshgrid(1:MM, 1:LL);
    center = [ceil(LL/2), ceil(MM/2)];
    radius = sqrt((x - center(2)).^2 + (y - center(1)).^2);
    fmSAC(radius > params.SidelobeCoeff) = 0;
end
%}

%% 显示PSF图像
waitbar(0.95, hWaitbar, '显示PSF图像...');

figure('Position', [50, 50, 1200, 900], 'Color', 'w');

if params.use_3d
    % 3D数据的中心切片显示和剖面图
    center_slice_xy = ceil(NN/2); % XY平面中心切片
    center_slice_xz = ceil(MM/2); % XZ平面中心切片（固定Y）
    center_slice_yz = ceil(LL/2); % YZ平面中心切片（固定X）
    
    % XY平面
    subplot(3,4,1);
    imagesc(squeeze(I_exc(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{exc} (XY Plan)', 'FontSize', 12);
    
    subplot(3,4,2);
    imagesc(squeeze(I_hexc(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{hexc} (XY Plan)', 'FontSize', 12);
    
    subplot(3,4,3);
    imagesc(squeeze(sig_fund(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Fundamental Freq (XY Plan)', 'FontSize', 12);
    
    subplot(3,4,4);
    imagesc(squeeze(sig_harm(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Harmonic Freq (XY Plan)', 'FontSize', 12);
    
    % XZ平面（固定Y）
    subplot(3,4,5);
    imagesc(rot90(squeeze(I_exc(:,center_slice_xz,:)), -1)); axis square; colorbar; % 顺时针旋转90度
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{exc} (XZ Plan)', 'FontSize', 12);
    
    subplot(3,4,6);
    imagesc(rot90(squeeze(I_hexc(:,center_slice_xz,:)), -1)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{hexc} (XZ Plan)', 'FontSize', 12);
    
    subplot(3,4,7);
    imagesc(rot90(squeeze(sig_fund(:,center_slice_xz,:)), -1)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Fundamental Freq (XZ Plan)', 'FontSize', 12);
    
    subplot(3,4,8);
    imagesc(rot90(squeeze(sig_harm(:,center_slice_xz,:)), -1)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Harmonic Freq (XZ Plan)', 'FontSize', 12);

    % fmSAC和SAC的显示
    subplot(3,4,9);
    imagesc(squeeze(fmSAC(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('fmSAC (XY Plan)', 'FontSize', 12);
    
    subplot(3,4,10);
    imagesc(squeeze(y_SAC(:,:,center_slice_xy))); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Conventional SAC (XY Plan)', 'FontSize', 12);
    
    subplot(3,4,11);
    imagesc(rot90(squeeze(fmSAC(:,center_slice_xz,:)), -1)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('fmSAC (XZ Plan)', 'FontSize', 12);
    
    subplot(3,4,12);
    imagesc(rot90(squeeze(y_SAC(:,center_slice_xz,:)), -1)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Conventional SAC (XZ Plan)', 'FontSize', 12);

else
    % 2D数据的完整显示
    subplot(2,3,1);
    imagesc(I_exc(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{exc}', 'FontSize', 12);
    
    subplot(2,3,2); 
    imagesc(I_hexc(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('I_{hexc}', 'FontSize', 12);
    
    subplot(2,3,3);
    imagesc(sig_fund(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Fundamental Frequency', 'FontSize', 12);
    
    subplot(2,3,4);
    imagesc(sig_harm(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Harmonic Frequency', 'FontSize', 12);
    
    subplot(2,3,5);
    imagesc(fmSAC(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('fmSAC', 'FontSize', 12);
    
    subplot(2,3,6);
    imagesc(y_SAC(:,:,26)); axis square; colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    % title('Conventional SAC', 'FontSize', 12);
end

colormap("hot");

%% 分子成像仿真
waitbar(0.98, hWaitbar, '进行分子成像仿真...');

% 设置物理尺寸参数（根据您的实际系统调整）
pixel_size = 50;  % nm/pixel，典型超分辨显微镜像素尺寸
axial_step = 50; % nm/层，Z轴步长

% 设置样品参数
sample_params = struct(...
    'num_molecules', 200, ...        % 随机分子数量
    'sim_num', 15, ...              % 微管数量
    'step_num', 50, ...             % 微管步数
    'forces', 0, ...               % 作用力
    'KT', 4.1, ...                 % 热力学参数
    'A', 1000, ...                 % 持久长度
    'l', 1, ...                    % 步长
    'sigma', 0.7, ...                % 高斯平滑参数
    'segment_size', 1.5, ...         % 微管段大小
    'pixel_size', pixel_size, ...  % 像素尺寸(nm)
    'axial_step', axial_step ...   % Z轴步长(nm)
);

% 选择样品类型和尺寸（现在使用物理尺寸）
%{
if params.use_3d
    sample_type = 'random_3d';  % 或 'random_3d'或'microtubule_3d'
    physical_size = [5, 5, 5];       % 物理尺寸 [μm]：5μm x 5μm x 2μm
    % 转换为像素尺寸
    dimensions = round(physical_size .* [1000, 1000, 1000] ./ [pixel_size, pixel_size, axial_step]);
    fprintf('3D样品物理尺寸: %.1fμm x %.1fμm x %.1fμm\n', physical_size);
    fprintf('对应像素尺寸: %d x %d x %d pixels\n', dimensions);
else
    sample_type = 'random_2d';  % 或 'random_2d'或'microtubule_2d'
    physical_size = [5, 5];          % 物理尺寸 [μm]：5μm x 5μm
    % 转换为像素尺寸
    dimensions = round(physical_size .* [1000, 1000] ./ [pixel_size, pixel_size]);
    fprintf('2D样品物理尺寸: %.1fμm x %.1fμm\n', physical_size);
    fprintf('对应像素尺寸: %d x %d pixels\n', dimensions);
end
%}
sample_type = 'random_3d';  % 或 'random_3d'或'microtubule_3d'
physical_size = [5, 5, 5];       % 物理尺寸 [μm]：5μm x 5μm x 2μm
% 转换为像素尺寸
dimensions = round(physical_size .* [1000, 1000, 1000] ./ [pixel_size, pixel_size, axial_step]);
fprintf('3D样品物理尺寸: %.1fμm x %.1fμm x %.1fμm\n', physical_size);
fprintf('对应像素尺寸: %d x %d x %d pixels\n', dimensions);
% 生成样品
s = generate_sample(sample_type, dimensions, sample_params);

% 卷积成像
% if params.use_3d
Iconf_sample = convn(s, I_exc, 'same');
y_SAC_sample = convn(s, y_SAC, 'same');
fmSAC_sample = convn(s, fmSAC, 'same');
%{
else
    Iconf_sample = conv2(s(:, :, params.slice_z), I_exc, 'same');
    y_SAC_sample = conv2(s(:, :, params.slice_z), y_SAC, 'same');
    fmSAC_sample = conv2(s(:, :, params.slice_z), fmSAC, 'same');
end
%}

% 显示成像结果
if params.use_3d
    figure('Position', [100, 100, 1000, 800], 'Color', 'w');
    center_slice_xy = ceil(size(s, 3)/2); % XY中心切片
    center_slice_xz = ceil(size(s, 2)/2); % XZ中心切片（固定Y）
    center_slice_yz = ceil(size(s, 1)/2); % YZ中心切片（固定X）
    
    % XY平面
    subplot(3,4,1);
    imagesc(s(:,:,center_slice_xy)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Microtubule (XY Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,2);
    imagesc(Iconf_sample(:,:,center_slice_xy)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Confocal (XY Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,3);
    imagesc(y_SAC_sample(:,:,center_slice_xy)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Conventional SAC (XY Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,4);
    imagesc(fmSAC_sample(:,:,center_slice_xy)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('fmSAC (XY Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % XZ平面
    subplot(3,4,5);
    imagesc(rot90(squeeze(s(:,center_slice_xz,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Microtubule (XZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,6);
    imagesc(rot90(squeeze(Iconf_sample(:,center_slice_xz,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Confocal (XZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,7);
    imagesc(rot90(squeeze(y_SAC_sample(:,center_slice_xz,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Conventional SAC (XZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,8);
    imagesc(rot90(squeeze(fmSAC_sample(:,center_slice_xz,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('fmSAC (XZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % YZ平面
    subplot(3,4,9);
    imagesc(rot90(squeeze(s(center_slice_yz,:,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Microtubule (YZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,10);
    imagesc(rot90(squeeze(Iconf_sample(center_slice_yz,:,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Confocal (YZ Plan)', 'FontSize', 12, 'FontWeight', 'bold'); 
    
    subplot(3,4,11);
    imagesc(rot90(squeeze(y_SAC_sample(center_slice_yz,:,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('Conventional SAC (YZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(3,4,12);
    imagesc(rot90(squeeze(fmSAC_sample(center_slice_yz,:,:)), -1)); axis square;
    set(gca, 'XTick', [], 'YTick', []);
    title('fmSAC (YZ Plan)', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(hot);
    % 对分子或微管样品进行3D显示
    volumeViewer(s);

else
    slice_z = [1,6,11,16,21,26,31,36,41,46,51]; % 切片显示
    for k = 1:length(slice_z)
        figure('Position', [100, 100, 1000, 800], 'Color', 'w');
        subplot(2,2,1);
        imagesc(s(:, :, slice_z(k))); axis square;
        set(gca, 'XTick', [], 'YTick', []);
        title('Microtubule Structure', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(2,2,2);
        imagesc(Iconf_sample(:, :, slice_z(k))); axis square;
        set(gca, 'XTick', [], 'YTick', []);
        title('Confocal Imaging', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(2,2,3);
        imagesc(y_SAC_sample(:, :, slice_z(k))); axis square;
        set(gca, 'XTick', [], 'YTick', []);
        title('Conventional SAC', 'FontSize', 14, 'FontWeight', 'bold');
        
        subplot(2,2,4);
        imagesc(fmSAC_sample(:, :, slice_z(k))); axis square;
        set(gca, 'XTick', [], 'YTick', []);
        title('fmSAC Imaging', 'FontSize', 14, 'FontWeight', 'bold');
        colormap("hot");
    end
end

%% 导出四个堆栈为 hot 伪彩 24-bit RGB TIFF（ImageJ 直接彩色）
fprintf('导出 hot 伪彩 RGB TIFF...\n');

stacks = {s, Iconf_sample, y_SAC_sample, fmSAC_sample};
names  = {'s_rgb', 'Iconf_rgb', 'y_SAC_rgb', 'fmSAC_rgb'};
% sli_z = [10,20,30,40,50,60,70,80,90,100];

% 预先生成 hot 色图（256 级）；也可以根据需要改成其他，比如parula等
hotMap = hot(256);          % 256×3  double 0-1

for k = 1:numel(stacks)
    img = stacks{k};                    % 取出矩阵
    gname = names{k};

    % 归一化到 0-255 索引
    img = single(img);
    img = img - min(img(:));
    img = img / max(img(:)) * 255;
    img = uint8(img);                   % 0-255 整数

    % 灰度 → RGB（hot）
    if ndims(img) == 3
        % ========= 3D：逐帧转 RGB =========
        [h, w, slices] = size(img);
        % t = Tiff(fname, 'w');
        % for i = 1:length(sli_z)
        for i = 1:slices
            % fname = gname + "_" + sli_z(i) + ".tif";
            fname = gname + "_" + i + ".tif";
            t = Tiff(fname, 'w');
            % idx = img(:, :, sli_z(i));
            idx = img(:, :, i);
            idx(idx==0) = 1;            % 避免 0 索引
            rgb = ind2rgb(idx, hotMap); % 0-1  double
            rgb = im2uint8(rgb);        % 0-255  uint8

            tag.ImageLength = h;
            tag.ImageWidth  = w;
            tag.BitsPerSample = 8;
            tag.SamplesPerPixel = 3;    % RGB
            tag.Photometric = Tiff.Photometric.RGB;
            tag.SampleFormat = Tiff.SampleFormat.UInt;
            tag.Compression = Tiff.Compression.None;
            tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            % tag.SubFileType = double(sli_z(i) > 1);
            tag.SubFileType = double(i > 1);
            setTag(t, tag);
            write(t, rgb);
            % if sli_z(i) < slices, writeDirectory(t); end
            if i < slices, writeDirectory(t); end
        end
        close(t);
    else
        % ========= 2D：单帧 =========
        idx = img;
        idx(idx==0) = 1;
        rgb = ind2rgb(idx, hotMap);
        imwrite(rgb, fname, 'Compression', 'none');
    end
    fprintf('  ✅ 已导出 %s（hot RGB）\n', fname);
end

fprintf('全部完成！拖进 ImageJ 即带 hot 颜色。\n');
% ------在imageJ中生成视频方法如下：---------------------------------------
% 打开imageJ，将tif拖进去
% image>stacks>3D projection
% 把rotation angle increment改小一些，比如改成5或者1，得到的视频会更细致
% File > Save As > AVI
% 如果想获得的视频更高清，需要一开始带入的mat文件是501×501
%% 完成
close(hWaitbar);
fprintf('仿真完成！\n');

%% 辅助函数 - 生成2D或3D荧光分子/微管分布
function sample = generate_sample(sample_type, dimensions, params)
% 生成荧光样品（随机分子或微管结构）
% 输入:
%   sample_type: 'random_2d', 'random_3d', 'microtubule_2d', 'microtubule_3d'
%   dimensions: [xsize, ysize] 或 [xsize, ysize, zsize]
%   params: 结构体，包含样品参数
%
% 输出:
%   sample: 生成的样品矩阵

    switch sample_type
        case 'random_2d'
            sample = generate_random_2d(dimensions, params);
        case 'random_3d'
            sample = generate_random_3d(dimensions, params);
        case 'microtubule_2d'
            sample = generate_microtubule_2d(dimensions, params);
        case 'microtubule_3d'
            sample = generate_microtubule_3d(dimensions, params);
        otherwise
            error('不支持的样品类型: %s', sample_type);
    end
end

function sample = generate_random_2d(dimensions, params)
    % 生成2D随机荧光分子
    n = dimensions(1); m = params.num_molecules;
    sample = zeros(2*n + 1, 2*n + 1);
    
    positions = rand(m, 2) .* (2*n);
    positions = ceil(positions);
    
    for i = 1:m
        x = min(max(positions(i, 1), 1), 2*n);
        y = min(max(positions(i, 2), 1), 2*n);
        
        if sample(x, y) == 0
            x_end = min(x + 1, 2*n);
            y_end = min(y + 1, 2*n);
            sample(x:x_end, y:y_end) = 1;
        end
    end
end

function sample = generate_random_3d(dimensions, params)
    % 生成3D随机荧光分子
    n = dimensions(1); m = params.num_molecules;
    sample = zeros(2*n+1, 2*n+1, 2*n+1);
    size_val = 3;
    
    coords = randperm((2*n+1)^3, m);
    [x, y, z] = ind2sub([2*n+1, 2*n+1, 2*n+1], coords);
    
    for i = 1:m
        x_range = x(i):min(x(i)+size_val-1, 2*n+1);
        y_range = y(i):min(y(i)+size_val-1, 2*n+1);
        z_range = z(i):min(z(i)+size_val-1, 2*n+1);
        sample(x_range, y_range, z_range) = 1;
    end
end

function sample = generate_microtubule_2d(dimensions, params)
    % 生成2D微管结构
    xsize = dimensions(1); ysize = dimensions(2);
    sim_num = params.sim_num; step_num = params.step_num;
    
    % 生成3D微管并提取2D切片
    microtubule_3d = generate_microtubule_3d([xsize, ysize, 1], params);
    sample = microtubule_3d(:, :, 1);
    
    % 可选: 添加2D特定的处理
    sample = imgaussfilt(sample, params.sigma);
end

function sample = generate_microtubule_3d(dimensions, params)
    % 生成3D微管结构 (优化版本)
    xsize = dimensions(1); ysize = dimensions(2); zsize = dimensions(3);
    sim_num = params.sim_num; step_num = params.step_num;
    
    % 调用WLC模型生成微管轨迹
    [wlcseries] = WLCmicrotubules_optimized(params.forces, params.KT, params.A, params.l, step_num, sim_num);
    
    % 创建3D图像
    sample = zeros(xsize, ysize, zsize);
    
    for i = 1:sim_num
        % 归一化坐标到图像尺寸
        coords = squeeze(wlcseries(:,:,i));
        coords_normalized = normalize_coordinates(coords, [xsize, ysize, zsize]);
        
        % 将轨迹点绘制到图像中
        for j = 1:size(coords_normalized, 1)
            x = round(coords_normalized(j, 1));
            y = round(coords_normalized(j, 2));
            z = round(coords_normalized(j, 3));
            
            if x >= 1 && x <= xsize && y >= 1 && y <= ysize && z >= 1 && z <= zsize
                % 绘制微管段（增加厚度）
                sample = draw_microtubule_segment(sample, [x, y, z], params.segment_size);
            end
        end
    end
    
    % 高斯平滑
    if zsize > 1
        sample = imgaussfilt3(sample, params.sigma);
    else
        sample = imgaussfilt(sample, params.sigma);
    end
end

function coords_normalized = normalize_coordinates(coords, target_size)
    % 归一化坐标到目标尺寸
    coords_normalized = coords - min(coords);
    coords_normalized = coords_normalized ./ max(coords_normalized(:));
    coords_normalized = coords_normalized .* (target_size - 1) + 1;
end

function img = draw_microtubule_segment(img, center, radius)
    % 绘制微管段（增加厚度）
    [x, y, z] = meshgrid(1:size(img,2), 1:size(img,1), 1:size(img,3));
    distances = sqrt((x - center(2)).^2 + (y - center(1)).^2 + (z - center(3)).^2);
    img(distances <= radius) = 1;
end

function [wlcseries] = WLCmicrotubules_optimized(forces, KT, A, l, steptot, sim_num)
    % 优化的WLC微管生成函数
    wlcseries = [];
    
    for ff = 1:length(forces)
        f = forces(ff);
        probmax = exp(f * l / KT);
        
        DNAseriestot = [];
        
        for sim_idx = 1:sim_num
            DNAt = [0, 0, 0];
            inirnd = rand(1, 3) * 2 - 1;
            DNAt = [DNAt; inirnd ./ sqrt(sum(inirnd.^2))];
            DNAseries = DNAt;
            
            indx = 3;
            for tt = 1:10000000
                dirtemp = (2 * rand(3, 1) - 1);
                direction = dirtemp / sqrt(sum(dirtemp.^2));
                costheta = direction(3);
                phi2 = 2 * (1 - DNAt(indx-1, :) * direction);
                prob = exp(f * l * costheta / KT - A / 2 / l * phi2);
                
                if rand * probmax < prob
                    DNAt = [DNAt; direction'];
                    DNAseries = [DNAseries; DNAseries(indx-1, :) + direction'];
                    indx = indx + 1;
                end
                
                if indx > steptot
                    break;
                end
            end
            
            DNAseriestot = cat(3, DNAseriestot, DNAseries);
        end
        
        wlcseries = cat(4, wlcseries, DNAseriestot);
    end
    
    if length(forces) == 1
        wlcseries = squeeze(wlcseries);
    end
end
elapsedTime = toc;
fprintf('代码运行时间为: %.4f 秒\n', elapsedTime);