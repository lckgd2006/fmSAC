%% 探讨alpha对fmSAC的影响 - 修复变量冲突版 Figure 18
clc
clear all
close all
addpath(genpath('PSF'));
addpath(genpath('CSV'));
tic;
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

I_s = 10e3;         % 对应光强10k W/cm2
I_d = 500e3;        % 抑制光强度

f1 = 10e3;          % f1频率
f2 = 15e3;          % f2频率
interval = 10e-6;
t = 0:interval:1-interval;
m_s = 1;            % 激发调制对比度CM=(Imax-Imin)/(Imax+Imin)
m_d = 0.9;          % 抑制光调制对比度

%% 加载PSF数据
a=load('I_exc532_51_3D.mat');
I1=a.result.PSF(:,:,25);
b=load('I_hexc488_51_3D.mat');
I2=b.result.PSF(:,:,25);

% 归一化并缩放
I1 = I1 / max(I1(:));
I2 = I2 / max(I2(:));
I_exc = I_s * I1;   % 对应光强10kW/cm2
I_hexc = I_d * I2;

% 获取数据尺寸并自动确定中心
[rows, cols] = size(I1);
center_col = round(cols/2); % 自动确定中心列

%% 初始化参数 - 重命名变量避免冲突
alpha_coeffs = 0:0.1:2.5; % 重命名：alpha_coeffs代替alpha
num_coeffs = length(alpha_coeffs);

% 预分配结果数组
FWHM = zeros(1, num_coeffs);
Neg_vals = zeros(1, num_coeffs); % 重命名：Neg_vals代替Neg
fmSAC_profiles = zeros(rows, num_coeffs);

% FFT相关参数
N = length(t);
frequencies = (-N/2:N/2-1) * (1/(N*interval));
f1_idx = find(abs(frequencies - f1) == min(abs(frequencies - f1)), 1);
f2_idx = find(abs(frequencies - f2) == min(abs(frequencies - f2)), 1);

%% 计算fmSAC的FWHM
fprintf('计算fmSAC...\n');
progressBar = waitbar(0, '🚀 计算进度: 0%', 'Name', 'α系数扫描');

for m = 1:num_coeffs
    current_alpha = alpha_coeffs(m); % 使用current_alpha避免冲突
    sig_fund = zeros(rows, 1);
    sig_harm = zeros(rows, 1);
    
    for i = 1:rows
        % 计算速率常数
        k_s = sigma_s * I_exc(i, center_col) * lambda_s / (h * c);
        k_d = sigma_d * I_hexc(i, center_col) * lambda_d / (h * c);
        
        % 双调制SAC信号
        y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
              (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);
        
        % FFT分析
        f_omiga = fft(y_s);
        f_omiga_shift = fftshift(f_omiga);
        result = abs(f_omiga_shift) / max(abs(f_omiga_shift));
        
        % 计算总功率（去除DC分量）
        total_power = (sum(result) - result(N/2+1))/2;
        
        % 提取特定频率分量
        sig_fund(i) = result(f1_idx) / total_power;
        sig_harm(i) = result(f2_idx) / total_power;
    end
    
    % 计算fmSAC信号
    fmSAC_signal = sig_fund - current_alpha * sig_harm;
    fmSAC_profiles(:, m) = fmSAC_signal;
    
    % 归一化
    fmSAC_signal = fmSAC_signal / max(fmSAC_signal);
    
    % 计算FWHM
    half_max = max(fmSAC_signal) / 2;
    half_index = find(fmSAC_signal >= half_max);
    FWHM(m) = length(half_index);
    
    % 记录最小值（负值）
    Neg_vals(m) = min(fmSAC_signal);
    
    % 更新进度条
    waitbar(m/num_coeffs, progressBar, sprintf('🚀 计算进度: %.0f%% (α=%.1f)', m/num_coeffs*100, current_alpha));
end

close(progressBar);

%% 三种评价方法计算
% 归一化FWHM和负值
normalized_FWHM = (FWHM - min(FWHM)) / (max(FWHM) - min(FWHM));
normalized_Neg = (abs(Neg_vals) - min(abs(Neg_vals))) / (max(abs(Neg_vals)) - min(abs(Neg_vals)));

% 方法1: 归一化加权和
weight_FWHM = 0.7;
weight_Neg = 0.3;
performance_metric1 = weight_FWHM * normalized_FWHM + weight_Neg * normalized_Neg;
[best_performance1, optimal_idx1] = min(performance_metric1);
optimal_alpha1 = alpha_coeffs(optimal_idx1);

% 方法2: 几何平均
performance_metric2 = sqrt(normalized_FWHM .* normalized_Neg);
[best_performance2, optimal_idx2] = min(performance_metric2);
optimal_alpha2 = alpha_coeffs(optimal_idx2);

% 方法3: 带惩罚项的指标
penalty = 1 + 0.5 * (abs(Neg_vals) > 0.1); % 负值超过0.1时惩罚
performance_metric3 = normalized_FWHM .* penalty;
[best_performance3, optimal_idx3] = min(performance_metric3);
optimal_alpha3 = alpha_coeffs(optimal_idx3);

% 记录各方法的最优结果
optimal_results = [
    optimal_alpha1, FWHM(optimal_idx1), Neg_vals(optimal_idx1), best_performance1;
    optimal_alpha2, FWHM(optimal_idx2), Neg_vals(optimal_idx2), best_performance2;
    optimal_alpha3, FWHM(optimal_idx3), Neg_vals(optimal_idx3), best_performance3
];

%% 创建炫酷的可视化 - 修复变量冲突版
fig=figure('Position', [50, 50, 1500, 950], 'Color', 'w', 'Name', 'The Effect of α Coefficient on fmSAC');

% 创建自定义颜色映射
cmap = jet(num_coeffs);

% 左上角：所有fmSAC剖面
% pos1 = [0.1 0.1 0.45 0.45];
subplot(2,2,1);
% Position = [left, bottom, width, height] (归一化坐标，范围0-1)
% ax1.Position = [0.5, 0.5, 0.5, 0.5];
hold on;

% 先绘制所有灰色线（不显示在图例中）
for m = 1:num_coeffs
    normalized_profile = fmSAC_profiles(:, m) / max(fmSAC_profiles(:, m));
    if m ~= optimal_idx1 && m ~= optimal_idx2 && m ~= optimal_idx3
        plot(0:(rows-1), normalized_profile, 'LineWidth', 1, 'Color', [0.7, 0.7, 0.7], ...
            'HandleVisibility', 'off'); % 关键：关闭这些线的图例显示
    end
end

% 绘制三种方法的最优线
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


% 右上角：FWHM vs Alpha (散点图+连线)
subplot(2, 2, 2);
% 创建单独的绘图句柄用于图例
h1 = scatter(alpha_coeffs, FWHM, 'filled', 'MarkerEdgeColor', [0.2, 0.6, 1], ...
    'DisplayName', 'FWHM data points');
hold on;
h2 = plot(alpha_coeffs, FWHM, 'Color', [0, 0, 0], 'LineWidth', 2, ...
    'DisplayName', 'FWHM Trendline'); % 使用纯黑色

% 标记三种方法的最优点
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
legend([h1, h2, h3, h4, h5], 'Location', 'northeast', 'FontSize', 9, 'Box','off'); % 明确指定图例内容

% 左下角：3D瀑布图显示所有剖面
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

% 右下角：负值 vs Alpha (面积图)
subplot(2, 2, 4);
% 使用area函数，它支持FaceAlpha属性
area_handle = area(alpha_coeffs, Neg_vals, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', [0.6, 0.1, 0.1], 'LineWidth', 2);
set(area_handle, 'FaceAlpha', 0.6); % area对象支持FaceAlpha
hold on;
plot(alpha_coeffs, zeros(size(alpha_coeffs)), 'k--', 'LineWidth', 2);

% 标记三种方法的最优点
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


% 添加总体标题
sgtitle(sprintf('The Effect of the α Coefficient on fmSAC Performance (CM₂=%.1f)', m_d), ...
    'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.4]);

%% 创建三种评价方法比较图 - 一行三列
figure('Position', [100, 100, 1500, 500], 'Color', 'w', 'Name', '三种评价方法比较');

% 子图1: 方法1 - 归一化加权和
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


% 子图2: 方法2 - 几何平均
subplot(1, 3, 2);
plot(alpha_coeffs, performance_metric2, 'b-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
hold on;
plot(optimal_alpha2, best_performance2, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
set(gca, 'LineWidth', 2, 'FontWeight', 'bold', 'FontSize', 12);
xlabel('α Coefficient', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Comprehensive Performance Metrics', 'FontWeight', 'bold', 'FontSize', 15);
title('Method 2: Geometric mean', 'FontWeight', 'bold', 'FontSize', 15);
grid on;
text(0.55, 0.95, sprintf('Optimal α = %.1f\nMetrics = %.3f\nFWHM = %d nm\nNegative Value = %.3f', ...
    optimal_alpha2, best_performance2, FWHM(optimal_idx2), Neg_vals(optimal_idx2)), ...
    'Units', 'normalized', 'FontWeight', 'bold', 'FontSize', 9, ...
    'BackgroundColor', 'NONE', 'VerticalAlignment', 'top');


% 子图3: 方法3 - 带惩罚项的指标
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


% 添加总体标题
sgtitle('Comparison of Three Evaluation Methods', 'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.1, 0.1, 0.4]);
% Lower values indicate better performance
%% 输出详细结果分析
fprintf('\n=== α系数优化结果分析 ===\n');
fprintf('📊 FWHM范围: [%d, %d] nm\n', min(FWHM), max(FWHM));
fprintf('📈 负值范围: [%.3f, %.3f]\n', min(Neg_vals), max(Neg_vals));

fprintf('\n=== 三种评价方法结果对比 ===\n');
fprintf('方法\t\t最优α\tFWHM(nm)\t负值\t\t指标值\n');
fprintf('----\t\t-----\t--------\t----\t\t------\n');
fprintf('加权和\t\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(1,1), optimal_results(1,2), optimal_results(1,3), optimal_results(1,4));
fprintf('几何平均\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(2,1), optimal_results(2,2), optimal_results(2,3), optimal_results(2,4));
fprintf('惩罚项\t\t%.1f\t%d\t\t%.3f\t\t%.3f\n', optimal_results(3,1), optimal_results(3,2), optimal_results(3,3), optimal_results(3,4));

% 分析各方法特点
fprintf('\n=== 方法特点分析 ===\n');
fprintf('🎯 方法1 (加权和): 可灵活调整权重，平衡FWHM和负值\n');
fprintf('📐 方法2 (几何平均): 对FWHM和负值同等敏感，要求两者都小\n');
fprintf('⚡ 方法3 (惩罚项): 对负值有硬性约束，适合负值敏感的应用\n');

% 推荐建议
[min_fwhm_idx] = find(FWHM == min(FWHM), 1);
[min_neg_idx] = find(abs(Neg_vals) == min(abs(Neg_vals)), 1);

fprintf('\n💡 推荐建议:\n');
fprintf('   如果分辨率最重要: 选择α=%.1f (FWHM=%d nm)\n', alpha_coeffs(min_fwhm_idx), min(FWHM));
fprintf('   如果伪像控制最重要: 选择α=%.1f (负值=%.3f)\n', alpha_coeffs(min_neg_idx), Neg_vals(min_neg_idx));
fprintf('   如果需要平衡考虑: 根据应用需求选择上述三种方法之一\n');

elapsedTime = toc;
fprintf('\n✨ 计算完成！\n');
fprintf('⏱️  代码运行时间为: %.4f 秒\n', elapsedTime);