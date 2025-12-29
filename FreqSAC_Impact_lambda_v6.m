%% SAC调制频谱分析系统 - 多染料波长参数影响研究（灵活版本）
% 可选择生成7个独立图表或1个组合图表
clc; clear; close all;

addpath(genpath('CSV'));
tic;

%% 绘图模式选择
% 设置为 'individual' 生成7个独立图表，设置为 'combined' 生成1个组合图表
plotMode = 'individual'; % 可修改为 'combined'或'individual'

%% 染料文件列表
dyeFiles = {
    '540-560orange.csv', 'Orange Dye (540-560)';
    '580-605red.csv', 'Red Dye (580-605)'; 
    '625-645crimson.csv', 'Crimson Dye (625-645)';
    '640-660deep red.csv', 'Deep Red Dye (640-660)';
    'Qdot 565.csv', 'Qdot 565';
    'Qdot 605.csv', 'Qdot 605';
    'Rhodamine 6G.csv', 'Rhodamine 6G'
};

numDyes = size(dyeFiles, 1);

%% 全局参数设置
globalParams = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'sigma_s_ref', 2.7e-16, ...  % 参考吸收截面 (532nm处)
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'm_s', 1 ...
);

%% 分析参数配置
analysisConfig = struct(...
    'I_s', 10e3, ...          % 激发光强度 (W/cm²)
    'I_d', 1000e3, ...        % 竞争光强度 (W/cm²)
    'f1', 10e3, ...           % 激发调制频率 (Hz)
    'f2', 15e3, ...           % 竞争调制频率 (Hz)
    'm_d', 0.6, ...           % 竞争调制对比度
    'duration', 0.1, ...      % 信号持续时间 (s)
    'interval', 10e-6, ...    % 采样间隔 (s)
    'lambda_s_fixed', 532, ... % 固定激发波长 (nm)
    'lambda_d_fixed', 561 ...  % 固定竞争波长 (nm)
);

%% 波长扫描范围
lambda_range = 480:5:680;

%% 辅助函数 - 读取染料数据
function [wavelengths, excitation, emission] = readDyeData(filename)
    fprintf('读取文件: %s\n', filename);
    try
        data = readmatrix(filename);
        if size(data, 2) >= 3
            wavelengths = data(:, 1);
            excitation = data(:, 2);
            emission = data(:, 3);
        else
            wavelengths = data(:, 1);
            excitation = data(:, 2);
            emission = zeros(size(excitation));
        end
        
        % 处理负值和异常值
        excitation(excitation < 0) = 0;
        emission(emission < 0) = 0;
        
        % 归一化激发谱和发射谱
        if max(excitation) > 0
            excitation = excitation / max(excitation);
        else
            excitation = zeros(size(excitation));
        end
        
        if max(emission) > 0
            emission = emission / max(emission);
        else
            emission = zeros(size(emission));
        end
        
        fprintf('数据范围: %d-%d nm, 激发谱最大值: %.4f, 发射谱最大值: %.4f\n', ...
            min(wavelengths), max(wavelengths), max(excitation), max(emission));
        
    catch ME
        fprintf('读取文件错误: %s\n', ME.message);
        wavelengths = 400:800;
        excitation = zeros(size(wavelengths));
        emission = zeros(size(wavelengths));
    end
end

%% 辅助函数 - 获取特定波长的激发谱值
function exc_value = getExcitationValue(wavelength, wavelengths, excitation_spectrum)
    if isempty(wavelengths) || isempty(excitation_spectrum)
        exc_value = 0;
        return;
    end
    
    if wavelength < min(wavelengths) || wavelength > max(wavelengths)
        exc_value = 0;
    else
        % 插值计算
        exc_value = interp1(wavelengths, excitation_spectrum, wavelength, 'linear', 0);
        if isnan(exc_value) || exc_value < 0 
            % 检查插值结果是否为 NaN（不是数字）或负值
            exc_value = 0;
        end
    end
end

%% 辅助函数 - 计算SAC信号和频谱分量（简化稳定版本）
function results = computeSACComponents(globalParams, I_s, I_d, f1, f2, m_d, ...
                                       duration, interval, lambda_s_nm, lambda_d_nm, ...
                                       wavelengths, excitation_spectrum)
    try
        % 转换波长单位：nm -> cm
        lambda_s = lambda_s_nm * 1e-7;
        lambda_d = lambda_d_nm * 1e-7;
        
        % 计算相关常数
        h = globalParams.h;
        c = globalParams.c;
        c1 = 1 + globalParams.k_isc/globalParams.k_t;
        
        % 获取激发谱值
        exc_value_s = getExcitationValue(lambda_s_nm, wavelengths, excitation_spectrum);
        exc_value_d = getExcitationValue(lambda_d_nm, wavelengths, excitation_spectrum);
        
        % 计算相对吸收截面
        ref_exc_value = max(excitation_spectrum);
        if ref_exc_value <= 0
            ref_exc_value = 1;
        end
        
        sigma_s = globalParams.sigma_s_ref * (exc_value_s / ref_exc_value);
        sigma_d = globalParams.sigma_s_ref * (exc_value_d / ref_exc_value);
        
        % 确保最小吸收截面
        min_sigma = globalParams.sigma_s_ref * 1e-6;
        sigma_s = max(sigma_s, min_sigma);
        sigma_d = max(sigma_d, min_sigma);
        
        % 计算激发和竞争速率
        k_s = sigma_s * I_s * lambda_s / (h * c);
        k_d = sigma_d * I_d * lambda_d / (h * c);
        
        % 生成时间序列
        t = 0:interval:duration-interval;
        n = length(t);
        
        % 生成调制信号
        cos_f1 = cos(2*pi*f1*t);
        cos_f2 = cos(2*pi*f2*t);
        
        numerator = k_s * (1 + globalParams.m_s * cos_f1);
        denominator = c1*(k_s*(1 + globalParams.m_s * cos_f1) + ...
                       k_d*(1 + m_d * cos_f2)) + globalParams.k0;
        
        % 避免除以0
        denominator(denominator == 0) = 1e-30;
        y_s = numerator ./ denominator;
        
        % 傅里叶变换分析
        f_fft = fft(y_s);
        f_fft_shift = fftshift(f_fft);
        result = abs(f_fft_shift) / max(abs(f_fft_shift));
        
        % 计算频率索引
        freq_resolution = (1/interval)/n;
        target_freqs = [f1, f2, f1+f2, abs(f2-f1), 2*f1, 3*f1];
        indices = round(target_freqs / freq_resolution) + floor(n/2) + 1;
        indices = min(max(indices, 1), n);
        
        % 计算频率分量占比
        dc_index = floor(n/2) + 1;
        total_power = (sum(result) - result(dc_index))/2;
        
        if total_power > 0
            fund = result(indices(1)) / total_power;
            harm = result(indices(2)) / total_power;
            sum_freq = result(indices(3)) / total_power;
            diff_freq = result(indices(4)) / total_power;
            double_freq = result(indices(5)) / total_power;
            triple_freq = result(indices(6)) / total_power;
        else
            fund = 0.01; harm = 0.01; sum_freq = 0.01; 
            diff_freq = 0.01; double_freq = 0.01; triple_freq = 0.01;
        end
        
        results = struct(...
            'fund', fund, 'harm', harm, 'sum', sum_freq, ...
            'diff', diff_freq, 'double', double_freq, 'triple', triple_freq, ...
            'sigma_s', sigma_s, 'sigma_d', sigma_d, 'success', true);
        
    catch
        % 如果计算失败，返回默认值
        results = struct(...
            'fund', 0.01, 'harm', 0.01, 'sum', 0.01, ...
            'diff', 0.01, 'double', 0.01, 'triple', 0.01, ...
            'sigma_s', globalParams.sigma_s_ref * 1e-6, ...
            'sigma_d', globalParams.sigma_s_ref * 1e-6, ...
            'success', false);
    end
end

%% 辅助函数 - 创建单个染料的独立图表
function createIndividualPlot(dyeName, lambda_range, components_s, components_d, ...
                             wavelengths, excitation, emission, labels, colors, lineStyles, lineWidths, dyeIdx)
    
    fig = figure('Position', [50, 100, 1800, 450], 'Color', 'w', ...
                'Name', sprintf('%s - SAC波长影响分析', dyeName));
    
    % 第1个子图: 激发波长影响
    subplot(1, 3, 1);
    ax1 = gca;
    % Position = [left, bottom, width, height] (归一化坐标，范围0-1)
    ax1.Position = [0.04 0.12 0.28 0.7];
    hold on;
    for i = 1:6
        plot(lambda_range, components_s(:, i), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, ...
            'LineWidth', lineWidths(i), 'DisplayName', labels{i});
    end
    hold off;
    grid on;
    set(gca, 'FontSize', 9, 'FontWeight', 'bold');
    xlabel('Excitation Wavelength λ_s (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Signal Intensity Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s\nFrequency Components vs Excitation Wavelength λ_s', ...
        dyeName), 'FontSize', 12, 'FontWeight', 'bold');
    xlim([450, 700]);
    ylim([0, 100]);
    legend('show','Location','northeast','FontSize',6,'Box','off');
    
    % 第2个子图: 竞争波长影响
    subplot(1, 3, 2);
    ax2 = gca;
    % Position = [left, bottom, width, height] (归一化坐标，范围0-1)
    ax2.Position = [0.36 0.12 0.28 0.7];
    hold on;
    for i = 1:6
        plot(lambda_range, components_d(:, i), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, ...
            'LineWidth', lineWidths(i), 'DisplayName', labels{i});
    end
    hold off;
    grid on;
    set(gca, 'FontSize', 9, 'FontWeight', 'bold');
    xlabel('Competition Wavelength λ_d (nm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Signal Intensity Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s\nFrequency Components vs Competition Wavelength λ_d', ...
        dyeName), 'FontSize', 12, 'FontWeight', 'bold');
    xlim([450, 700]);
    ylim([0, 100]);
    legend('show','Location','northeast','FontSize',6,'Box','off');
        
    % 第3个子图: 激发和发射光谱
    subplot(1, 3, 3);
    ax3 = gca;
    % Position = [left, bottom, width, height] (归一化坐标，范围0-1)
    ax3.Position = [0.68 0.12 0.28 0.7];        
    % 确定有效波长范围
    if ~isempty(wavelengths)
        valid_idx = wavelengths >= 400 & wavelengths <= 800;
        if sum(valid_idx) > 10  % 至少有10个有效数据点
            wavelengths_plot = wavelengths(valid_idx);
            excitation_plot = excitation(valid_idx);
            emission_plot = emission(valid_idx);

            grid on;
            set(gca, 'FontSize', 9, 'FontWeight', 'bold');
            yyaxis left
            plot(wavelengths_plot, excitation_plot, 'b-', 'LineWidth', 2, 'DisplayName', 'Excitation Spectrum');
            ylabel('Normalized Excitation Intensity (a.u.)', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');
            ylim([0, 1]);

            yyaxis right
            plot(wavelengths_plot, emission_plot, 'r-', 'LineWidth', 2, 'DisplayName', 'Emission Spectrum');
            ylabel('Normalized Emission Intensity (a.u.)', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
            ylim([0, 1]);

            xlabel('Wavelength (nm)', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('%s\n Excitation / Emission Spectrum', dyeName), 'FontSize', 12, 'FontWeight', 'bold');
            xlim([400, 800]);
            legend({'Excitation Spectrum','Emission Spectrum'},'Location','northeast','FontSize',6,'Box','off');
        else
            % 数据不足，显示提示
            text(0.5, 0.5, '光谱数据不足', 'HorizontalAlignment', 'center', ...
                'FontSize', 16, 'FontWeight', 'bold');
            xlim([0, 1]);
            ylim([0, 1]);
            set(gca, 'XTick', [], 'YTick', []);
            title(sprintf('%s\n光谱数据', dyeName), 'FontSize', 14, 'FontWeight', 'bold');
        end
    else
        % 无数据情况
        text(0.5, 0.5, '无光谱数据', 'HorizontalAlignment', 'center', ...
            'FontSize', 15, 'FontWeight', 'bold');
        xlim([0, 1]);
        ylim([0, 1]);
        set(gca, 'XTick', [], 'YTick', []);
        title(sprintf('%s\n光谱数据', dyeName), 'FontSize', 14, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
 
    % 调整子图间距
    sgtitle(sprintf('Dye: %s - fmSAC Modulation Spectrum Analysis', dyeName), ...
           'FontSize', 15, 'FontWeight', 'bold', 'Color', 'blue');
    
    % 保存图表（可选）
    filename = sprintf('SAC_Analysis_%s.png', regexprep(dyeName, '[^a-zA-Z0-9]', '_'));
    saveas(fig, filename);
    fprintf('图表已保存: %s\n', filename);
end

%% 辅助函数 - 创建组合图表
function createCombinedPlot(allResults, labels, colors, lineStyles, lineWidths)
    % 创建大图
    figure('Position', [50, 50, 1800, 2500], 'Color', 'w', 'Name', '多染料SAC波长影响分析');
    
    for dyeIdx = 1:length(allResults)
        results = allResults{dyeIdx};
        
        % 第1列: 激发波长影响
        subplot(7, 3, (dyeIdx-1)*3 + 1);
        hold on;
        for i = 1:6
            plot(results.lambda_s_analysis.wavelengths, results.lambda_s_analysis.components(:,i), ...
                'Color', colors(i,:), 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i));
        end
        hold off;
        grid on;
        ylabel('信号强度比 (%)', 'FontSize', 9, 'FontWeight', 'bold');
        if dyeIdx == 7
            xlabel('激发波长 λ_s (nm)', 'FontSize', 10, 'FontWeight', 'bold');
        end
        title(sprintf('%s\n激发波长影响', results.name), 'FontSize', 10, 'FontWeight', 'bold');
        ylim([0, 100]);
        
        % 第2列: 竞争波长影响
        subplot(7, 3, (dyeIdx-1)*3 + 2);
        hold on;
        for i = 1:6
            plot(results.lambda_d_analysis.wavelengths, results.lambda_d_analysis.components(:,i), ...
                'Color', colors(i,:), 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i));
        end
        hold off;
        grid on;
        if dyeIdx == 7
            xlabel('竞争波长 λ_d (nm)', 'FontSize', 10, 'FontWeight', 'bold');
        end
        title(sprintf('%s\n竞争波长影响', results.name), 'FontSize', 10, 'FontWeight', 'bold');
        ylim([0, 100]);
        
        % 第3列: 激发和发射光谱
        subplot(7, 3, (dyeIdx-1)*3 + 3);
        wavelengths_spectra = results.spectra.wavelengths;
        excitation_spectra = results.spectra.excitation;
        emission_spectra = results.spectra.emission;
        
        % 找到有效数据范围
        valid_idx = wavelengths_spectra >= 400 & wavelengths_spectra <= 800;
        if sum(valid_idx) > 0
            wavelengths_plot = wavelengths_spectra(valid_idx);
            excitation_plot = excitation_spectra(valid_idx);
            emission_plot = emission_spectra(valid_idx);
            
            yyaxis left
            plot(wavelengths_plot, excitation_plot, 'b-', 'LineWidth', 2);
            ylabel('激发谱', 'FontSize', 9, 'Color', 'b', 'FontWeight', 'bold');
            ylim([0, 1]);
            
            yyaxis right
            plot(wavelengths_plot, emission_plot, 'r-', 'LineWidth', 2);
            ylabel('发射谱', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold');
            ylim([0, 1]);
            
            xlabel('波长 (nm)', 'FontSize', 10, 'FontWeight', 'bold');
            title(sprintf('%s\n激发/发射光谱', results.name), 'FontSize', 10, 'FontWeight', 'bold');
            grid on;
            xlim([450, 700]);
        else
            % 如果没有有效数据，显示空白图
            text(0.5, 0.5, '无光谱数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
            xlim([0, 1]);
            ylim([0, 1]);
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
    
    % 添加图例（可选）
    % subplot(7, 3, 20);
    % for i = 1:6
    %     plot(NaN, NaN, 'Color', colors(i,:), 'LineStyle', lineStyles{i}, ...
    %          'LineWidth', lineWidths(i), 'DisplayName', labels{i});
    % end
    % legend('show', 'Location', 'north', 'Orientation', 'horizontal', ...
    %        'FontSize', 8, 'NumColumns', 3);
    % axis off;
    
    % 保存组合图（可选）
    % saveas(gcf, 'MultiDye_SAC_Analysis_Combined.png');
    % fprintf('组合图已保存为 MultiDye_SAC_Analysis_Combined.png\n');
end

%% 主分析循环
fprintf('开始分析%d种染料...\n', numDyes);
fprintf('绘图模式: %s\n', plotMode);

% 频谱分量标签和样式
labels = {'ξ(f_1)', 'ξ(f_2)', 'ξ(f_1+f_2)', 'ξ(|f_1-f_2|)', 'ξ(2f_1)', 'ξ(3f_1)'};
colors = lines(6);
lineStyles = {'-', '--', ':', '-.', '-', '--'};
lineWidths = [2.5, 2.0, 2.0, 1.8, 1.8, 1.8];

% 存储所有结果（用于组合图）
allResults = cell(numDyes, 1);

for dyeIdx = 1:numDyes
    fprintf('\n=== 分析染料 %d/%d: %s ===\n', dyeIdx, numDyes, dyeFiles{dyeIdx, 2});
    
    % 读取染料数据
    [wavelengths, excitation, emission] = readDyeData(dyeFiles{dyeIdx, 1});
    
    % 分析1: lambda_s扫描，lambda_d固定
    fprintf('进行激发波长扫描分析...\n');
    components_lambda_s = zeros(length(lambda_range), 6);
    
    for i = 1:length(lambda_range)
        results = computeSACComponents(globalParams, analysisConfig.I_s, analysisConfig.I_d, ...
            analysisConfig.f1, analysisConfig.f2, analysisConfig.m_d, analysisConfig.duration, ...
            analysisConfig.interval, lambda_range(i), analysisConfig.lambda_d_fixed, ...
            wavelengths, excitation);
        
        components_lambda_s(i,:) = [results.fund, results.harm, results.sum, ...
                                   results.diff, results.double, results.triple];
        
        if mod(i, 20) == 0
            fprintf('进度: %d/%d\n', i, length(lambda_range));
        end
    end
    
    % 分析2: lambda_s固定，lambda_d扫描
    fprintf('进行竞争波长扫描分析...\n');
    components_lambda_d = zeros(length(lambda_range), 6);
    
    for i = 1:length(lambda_range)
        results = computeSACComponents(globalParams, analysisConfig.I_s, analysisConfig.I_d, ...
            analysisConfig.f1, analysisConfig.f2, analysisConfig.m_d, analysisConfig.duration, ...
            analysisConfig.interval, analysisConfig.lambda_s_fixed, lambda_range(i), ...
            wavelengths, excitation);
        
        components_lambda_d(i,:) = [results.fund, results.harm, results.sum, ...
                                   results.diff, results.double, results.triple];
        
        if mod(i, 20) == 0
            fprintf('进度: %d/%d\n', i, length(lambda_range));
        end
    end
    
    % 存储结果（用于组合图）
    allResults{dyeIdx} = struct(...
        'name', dyeFiles{dyeIdx, 2}, ...
        'spectra', struct('wavelengths', wavelengths, 'excitation', excitation, 'emission', emission), ...
        'lambda_s_analysis', struct('wavelengths', lambda_range, 'components', components_lambda_s * 100), ...
        'lambda_d_analysis', struct('wavelengths', lambda_range, 'components', components_lambda_d * 100) ...
    );
    
    % 根据模式选择绘图方式
    if strcmp(plotMode, 'individual')
        % 为当前染料创建独立图表
        createIndividualPlot(dyeFiles{dyeIdx, 2}, lambda_range, ...
            components_lambda_s * 100, components_lambda_d * 100, ...
            wavelengths, excitation, emission, labels, colors, lineStyles, lineWidths, dyeIdx);
    end
    
    fprintf('完成 %s 的分析\n', dyeFiles{dyeIdx, 2});
end

% 如果选择组合模式，创建组合图表
if strcmp(plotMode, 'combined')
    fprintf('\n=== 生成组合图 ===\n');
    createCombinedPlot(allResults, labels, colors, lineStyles, lineWidths);
end

%% 保存分析结果（可选）
% analysisResults = struct(...
%     'allResults', {allResults}, ...
%     'globalParams', globalParams, ...
%     'analysisConfig', analysisConfig, ...
%     'dyeFiles', {dyeFiles}, ...
%     'timestamp', datetime('now') ...
% );
% 
% save('MultiDye_SAC_Wavelength_Analysis.mat', 'analysisResults');
% fprintf('分析结果已保存到 MultiDye_SAC_Wavelength_Analysis.mat\n');

fprintf('\n===== 分析完成 =====\n');
elapsedTime = toc;
fprintf('总运行时间: %.2f 秒\n', elapsedTime);