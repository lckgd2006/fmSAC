%% SAC调制频谱分析系统 - 多参数影响研究
% 研究CM2、I_exc、I_hexc、f1、f2等参数对频谱分量的影响
clc; clear; close all;

%% 全局参数设置
globalParams = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'sigma_s', 2.7e-16, ...
    'lambda_s', 532e-7, ...
    'lambda_d', 561e-7, ...
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'sigma_d_ratio', 0.049850201 ...
);

%% 分析参数配置
analysisConfig = struct(...
    'I_s', 10e3, ...          % 激发光强度 (W/cm²)
    'I_d', 500e3, ...        % 竞争光强度 (W/cm²)
    'f1', 10e3, ...           % 激发调制频率 (Hz)
    'f2', 15e3, ...           % 竞争调制频率 (Hz)
    'm_s', 1.0, ...           % 激发调制对比度
    'm_d', 1.0, ...           % 竞争调制对比度
    'duration', 1, ...        % 信号持续时间 (s)
    'interval', 10e-6, ...    % 采样间隔 (s)
    'high_res_interval', 0.1e-6 ... % 高分辨率采样间隔
);

%% 辅助函数 - 计算SAC信号和频谱分量（优化版）
function results = computeSACComponents(globalParams, I_s, I_d, f1, f2, m_s, m_d, duration, interval)
    % 计算相关常数
    h = globalParams.h;
    c = globalParams.c;
    c1 = 1 + globalParams.k_isc/globalParams.k_t;
    sigma_d = globalParams.sigma_s * globalParams.sigma_d_ratio;
    
    % 计算激发和竞争速率
    k_s = globalParams.sigma_s * I_s * globalParams.lambda_s / (h * c);
    k_d = sigma_d * I_d * globalParams.lambda_d / (h * c);
    
    % 生成时间序列（向量化计算）
    t = 0:interval:duration-interval;
    n = length(t);
    
    % 生成调制信号（向量化计算）
    cos_f1 = cos(2*pi*f1*t);
    cos_f2 = cos(2*pi*f2*t);
    
    numerator = k_s * (1 + m_s * cos_f1);
    denominator = c1*(k_s*(1 + m_s * cos_f1) + ...
                   k_d*(1 + m_d * cos_f2)) + globalParams.k0;
    y_s = numerator ./ denominator;
    
    % 傅里叶变换分析
    f_fft = fft(y_s);
    f_fft_shift = fftshift(f_fft);
    result = abs(f_fft_shift) / max(abs(f_fft_shift));
    
    % 计算频率索引
    freq_resolution = (1/interval)/n;
    indices = round([f1, f2, f1+f2, abs(f2-f1), 2*f1, 3*f1] / freq_resolution) + n/2 + 1;
    
    % 确保索引在有效范围内
    indices = min(max(indices, 1), n);
    
    % 计算频率分量占比
    sumx = (sum(result) - result(n/2+1)) / 2; % 排除直流分量
    if f1 == f2
        results = struct(...
            'fund', result(indices(1)) / sumx, ...
            'harm', result(indices(2)) / sumx, ...
            'sum', result(indices(3)) / sumx, ...
            'diff', result(indices(4)-1) / sumx, ...
            'double', result(indices(5)) / sumx, ...
            'triple', result(indices(6)) / sumx ...
        );
    else
        results = struct(...
            'fund', result(indices(1)) / sumx, ...
            'harm', result(indices(2)) / sumx, ...
            'sum', result(indices(3)) / sumx, ...
            'diff', result(indices(4)) / sumx, ...
            'double', result(indices(5)) / sumx, ...
            'triple', result(indices(6)) / sumx ...
        );
    end
end

%% 辅助函数 - 创建专业图表（优化版）
function createProfessionalPlot(xData, yData, labels, xLabel, yLabel, titleText, legendText)
    figure('Position', [100, 100, 1000, 700], 'Color', 'w', 'Name', titleText);
    
    colors = lines(size(yData, 2));
    lineStyles = {'-', '--', ':', '-.', '-', '--'};
    lineWidths = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
    markers = {'o', 's', '^', 'd', 'v', '+'};
    
    hold on;
    for i = 1:size(yData, 2)
        if length(xData) <= 10 % 数据点少时使用标记
            plot(xData, yData(:,i), ...
                'Color', colors(i,:), ...
                'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1}, ...
                'LineWidth', lineWidths(i), ...
                'Marker', markers{mod(i-1, length(markers)) + 1}, ...
                'MarkerSize', 8, ...
                'MarkerFaceColor', colors(i,:), ...
                'DisplayName', labels{i});
        else
            plot(xData, yData(:,i), ...
                'Color', colors(i,:), ... 
                'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1}, ...
                'LineWidth', lineWidths(i), ...
                'DisplayName', labels{i});
        end
    end
    hold off;
    
    % 设置图形属性
    grid on;
    % grid minor;
    ylim([0, 100]);

    set(gca, 'LineWidth', 2, 'FontSize', 18, 'FontWeight', 'bold', ...
             'XMinorTick', 'on', 'YMinorTick', 'on', ...
             'TickLength', [0.02, 0.02]);
    
    xlabel(xLabel, 'FontSize', 24, 'FontWeight', 'bold');
    ylabel(yLabel, 'FontSize', 24, 'FontWeight', 'bold');
    title(titleText, 'FontSize', 24, 'FontWeight', 'bold');
    
    legend('show', 'Location', 'northeast', 'FontSize', 12, 'Box', 'off');
    set(gcf, 'Color', 'w');
end

%% 进度显示函数
function showProgress(current, total, message)
    if mod(current, ceil(total/10)) == 0 || current == total
        fprintf('%s: %.0f%%\n', message, current/total*100);
    end
end

%% 1. CM1变化对频谱分量的影响
fprintf('=== 分析1: CM1调制深度对频谱分量的影响 ===\n');
m_s_range = 0:0.01:1;
components_relative_1 = zeros(length(m_s_range), 6);

for i = 1:length(m_s_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,analysisConfig.I_d,...
        analysisConfig.f1,analysisConfig.f2,m_s_range(i),analysisConfig.m_d,analysisConfig.duration,analysisConfig.interval);
    
    components_relative_1(i,:) = [results.fund, results.sum, results.diff, ...
                               results.double, results.triple, results.harm];
    
    showProgress(i, length(m_s_range), 'CM1分析进度');
end

labels = {'ξ(f_1)', 'ξ(f_1+f_2)', 'ξ(f_1-f_2)', 'ξ(2f_1)', 'ξ(3f_1)', 'ξ(f_2)'};
createProfessionalPlot(m_s_range*100, components_relative_1*100, labels, ...
    'CM_1 Modulation Depth (%)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs CM_1 Modulation Depth', labels);

%% 2. CM2变化对频谱分量的影响
fprintf('=== 分析2: CM2调制深度对频谱分量的影响 ===\n');
m_d_range = 0:0.01:1;
components_relative_2 = zeros(length(m_d_range), 6);

for i = 1:length(m_d_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,analysisConfig.I_d,...
        analysisConfig.f1,analysisConfig.f2,analysisConfig.m_s,m_d_range(i),analysisConfig.duration,analysisConfig.interval);
    
    components_relative_2(i,:) = [results.fund, results.sum, results.diff, ...
                               results.double, results.triple, results.harm];
    
    showProgress(i, length(m_d_range), 'CM2分析进度');
end

labels = {'ξ(f_1)', 'ξ(f_1+f_2)', 'ξ(f_1-f_2)', 'ξ(2f_1)', 'ξ(3f_1)', 'ξ(f_2)'};
createProfessionalPlot(m_d_range*100, components_relative_2*100, labels, ...
    'CM_2 Modulation Depth (%)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs CM_2 Modulation Depth', labels);

%% 3. 激发光强I_exc变化的影响
fprintf('\n=== 分析3: 激发光强对频谱分量的影响 ===\n');

I_exc_range = linspace(0, 300, 1001)*1e3; % 更合理的点数
components_exc = zeros(length(I_exc_range), 6);

for i = 1:length(I_exc_range)
    results = computeSACComponents(globalParams,I_exc_range(i), ...
        analysisConfig.I_d,analysisConfig.f1,analysisConfig.f2, ...
        analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
        analysisConfig.interval);
    
    components_exc(i,:) = [results.fund, results.sum, results.diff, ...
                          results.double, results.triple, results.harm];
    
    showProgress(i, length(I_exc_range), 'I_exc分析进度');
end

createProfessionalPlot(I_exc_range/1e3, components_exc*100, labels, ...
    'I_{exc} (kW/cm^2)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Excitation Intensity', labels);

%% 4. 竞争光强I_hexc变化的影响
fprintf('\n=== 分析4: 竞争光强对频谱分量的影响 ===\n');

% I_hexc_range = logspace(2, 6, 50); % 对数间隔，更好覆盖大范围
I_hexc_range = linspace(0, 1000, 1001)*1e3; % 更合理的点数
components_hexc = zeros(length(I_hexc_range), 6);

for i = 1:length(I_hexc_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s,...
        I_hexc_range(i),analysisConfig.f1,analysisConfig.f2,...
        analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
        analysisConfig.interval);
    
    components_hexc(i,:) = [results.fund, results.sum, results.diff, ...
                          results.double, results.triple, results.harm];
    
    showProgress(i, length(I_hexc_range), 'I_hexc分析进度');
end

createProfessionalPlot(I_hexc_range/1e6, components_hexc*100, labels, ...
    'I_{hexc} (MW/cm^2)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Competition Intensity', labels);

%% 5. 调制频率f1和f2变化的影响
fprintf('\n=== 分析5: 调制频率对频谱分量的影响 ===\n');

% f1频率变化的影响
% f1_range = [1, 5, 10, 20, 50, 100, 200, 400] * 1e3;
f1_range = linspace(1, 100, 100)*1e3; % 更合理的点数
components_f1 = zeros(length(f1_range), 6);

for i = 1:length(f1_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s, ...
        analysisConfig.I_d,f1_range(i),analysisConfig.f2,analysisConfig.m_s, ...
        analysisConfig.m_d,analysisConfig.duration,analysisConfig.high_res_interval);

    components_f1(i,:) = [results.fund, results.sum, results.diff, ...
        results.double, results.triple, results.harm];


    showProgress(i, length(f1_range), 'f1频率分析进度');
end

% f2频率变化的影响
% f2_range = [1, 5, 10, 20, 50, 100, 200, 400] * 1e3;
f2_range = linspace(1, 100, 100)*1e3; % 更合理的点数
components_f2 = zeros(length(f2_range), 6);

for i = 1:length(f2_range)
    results = computeSACComponents(globalParams,analysisConfig.I_s, ...
        analysisConfig.I_d,analysisConfig.f1,f2_range(i),analysisConfig.m_s, ...
        analysisConfig.m_d,analysisConfig.duration,analysisConfig.high_res_interval);
    
    components_f2(i,:) = [results.fund, results.sum, results.diff, ...
                         results.double, results.triple, results.harm];
    
    showProgress(i, length(f2_range), 'f2频率分析进度');
end

% 绘制频率影响结果
createProfessionalPlot(f1_range/1e3, components_f1*100, labels, ...
    'f_1 (kHz)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Excitation Modulation Frequency f_1', labels);

createProfessionalPlot(f2_range/1e3, components_f2*100, labels, ...
    'f_2 (kHz)', 'Signal Intensity Ratio (%)', ...
    'Frequency Components vs Competition Modulation Frequency f_2', labels);

%% 6. 频率组合优化分析
fprintf('\n=== 分析6: 频率组合优化分析 ===\n');

% f1_test_range = [1, 5, 10, 20, 50, 100] * 1e3;
% f2_test_range = [1, 5, 10, 20, 50, 100] * 1e3;
f1_test_range = linspace(1, 100, 10)*1e3;
f2_test_range = linspace(1, 100, 10)*1e3;
optimization_matrix = zeros(length(f1_test_range), length(f2_test_range));

total_points = length(f1_test_range) * length(f2_test_range);
current_point = 0;

for i = 1:length(f1_test_range)
    for j = 1:length(f2_test_range)
        results = computeSACComponents(globalParams,analysisConfig.I_s, ...
            analysisConfig.I_d,f1_test_range(i),f2_test_range(j), ...
            analysisConfig.m_s,analysisConfig.m_d,analysisConfig.duration, ...
            analysisConfig.high_res_interval);
        
        optimization_matrix(i,j) = results.fund * results.harm * 10000;
        
        current_point = current_point + 1;
        showProgress(current_point, total_points, '频率组合优化进度');
    end
end

% 绘制优化热图
figure('Position', [100, 100, 800, 800], 'Color', 'w');
imagesc(f2_test_range/1e3, f1_test_range/1e3, optimization_matrix);
colorbar;
set(gca, 'YDir', 'normal', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('f_2 (kHz)', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('f_1 (kHz)', 'FontSize', 24, 'FontWeight', 'bold');
title('Frequency Combination Optimization (ξ(f_1)×ξ(f_2))', 'FontSize',24, 'FontWeight', 'bold');
colormap(jet);
set(gcf, 'Color', 'w');

%% 7. 性能指标计算和结果保存
fprintf('\n=== 性能指标计算和结果保存 ===\n');

% 计算关键性能指标
performanceMetrics = struct(...
    'max_CM1_performance', max(components_relative_1(:)), ...
    'optimal_CM1', m_s_range(components_relative_1(:,1) == max(components_relative_1(:,1))), ...
    'max_CM2_performance', max(components_relative_2(:)), ...
    'optimal_CM2', m_d_range(components_relative_2(:,1) == max(components_relative_2(:,1))), ...
    'optimal_I_exc', I_exc_range(components_exc(:,1) == max(components_exc(:,1))) / 1e3, ...
    'optimal_I_hexc', I_hexc_range(components_hexc(:,1) == max(components_hexc(:,1))) / 1e6, ...
    'optimal_f1', f1_range(components_f1(:,1) == max(components_f1(:,1))) / 1e3, ...
    'optimal_f2', f2_range(components_f2(:,1) == max(components_f2(:,1))) / 1e3 ...
);

% 保存结果
analysisResults = struct(...
    'CM1_analysis', components_relative_1, ...
    'CM2_analysis', components_relative_2, ...
    'I_exc_analysis', components_exc, ...
    'I_hexc_analysis', components_hexc, ...
    'f1_analysis', components_f1, ... 
    'f2_analysis', components_f2, ...
    'frequency_optimization', optimization_matrix, ...
    'globalParams', globalParams, ...
    'analysisConfig', analysisConfig, ...
    'performanceMetrics', performanceMetrics, ...
    'timestamp', datetime('now') ...
);

% save('SAC_Analysis_Results.mat', 'analysisResults');
% fprintf('分析结果已保存到 SAC_Analysis_Results.mat\n');

%% 8. 生成总结报告
fprintf('\n===== SAC调制频谱分析总结 =====\n');
fprintf('分析完成时间: %s\n', datestr(now));
fprintf('总分析数据点: %d\n', length(m_s_range)+length(m_d_range) + ...
    length(I_exc_range) + length(I_hexc_range) + length(f1_range) + ...
    length(f2_range) + numel(optimization_matrix));
fprintf('\n主要发现:\n');
fprintf('1. 最佳CM1调制深度: %.2f\n', performanceMetrics.optimal_CM1);
fprintf('2. 最佳CM2调制深度: %.2f\n', performanceMetrics.optimal_CM2);
fprintf('3. 最佳激发光强: %.1f kW/cm²\n', performanceMetrics.optimal_I_exc);
fprintf('4. 最佳竞争光强: %.1f MW/cm²\n', performanceMetrics.optimal_I_hexc); 
fprintf('5. 最佳调制频率 f1: %.1f kHz\n', performanceMetrics.optimal_f1);
fprintf('6. 最佳调制频率 f2: %.1f kHz\n', performanceMetrics.optimal_f2);
fprintf('7. 最大性能指标: %.4f\n', performanceMetrics.max_CM2_performance);

fprintf('\n分析完成！所有结果已保存并可视化。\n');