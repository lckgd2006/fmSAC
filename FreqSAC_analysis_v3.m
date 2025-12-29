function freq_components = FreqSAC_analy(k_isc, ...
    k_t, k0, sigma_s, I_s, I_d, lambda_s, lambda_d, f1, f2, m_s, m_d, duration, interval)
% FreqSAC_analysis 分析双调制SAC信号的频谱特性
%
% 输入参数:
%   k_isc      - 系间窜越速率 (默认: 1.1e6)
%   k_t        - 三重态衰减速率 (默认: 0.49e6)
%   k0         - 荧光衰减速率 (默认: 2.56e8)
%   sigma_s    - 激发光吸收截面 (默认: 2.7e-16)
%   I_s        - 激发光强度 (W/cm²) (默认: 10e3)
%   I_d        - 竞争光强度 (W/cm²) (默认: 1000e3)
%   lambda_s   - 激发光波长 (cm) (默认: 532e-7)
%   lambda_d   - 竞争光波长 (cm) (默认: 561e-7)
%   f1         - 激发光调制频率 (Hz) (默认: 10e3)
%   f2         - 竞争光调制频率 (Hz) (默认: 15e3)
%   m_s        - 激发调制对比度 (默认: 1)
%   m_d        - 竞争光调制对比度 (默认: 0.5)
%   duration   - 信号持续时间 (s) (默认: 1)
%   interval   - 采样间隔 (s) (默认: 0.5e-6)
%
% 输出参数:
%   freq_components - 结构体，包含各频率分量的占比
%   omiga_shift    - 频率轴数据 (Hz)
%   result          - 归一化频谱幅度

% 设置默认参数
if nargin < 14
    interval = 0.5e-6;
end
if nargin < 13
    duration = 1;
end
if nargin < 12
    m_d = 0.5;
end
if nargin < 11
    m_s = 1;
end
if nargin < 10
    f2 = 15e3;
end
if nargin < 9
    f1 = 10e3;
end
if nargin < 8
    lambda_d = 561e-7;
end
if nargin < 7
    lambda_s = 532e-7;
end
if nargin < 6
    I_d = 1000e3;
end
if nargin < 5
    I_s = 10e3;
end
if nargin < 4
    sigma_s = 2.7e-16;
end
if nargin < 3
    k0 = 2.56e8;
end
if nargin < 2
    k_t = 0.49e6;
end
if nargin < 1
    k_isc = 1.1e6;
end

% 计算相关常数
h = 6.626e-34;      % 普朗克常数
c = 3e10;           % 光速 (cm/s)
c1 = 1 + k_isc/k_t;

% 计算损耗光吸收截面 (基于R6G染料的比例关系)
sigma_d = sigma_s * 0.049850201;

% 计算激发和损耗速率
k_s = sigma_s * I_s * lambda_s / (h * c);
k_d = sigma_d * I_d * lambda_d / (h * c);

% 生成时间序列
t = 0:interval:duration-interval;

% 生成调制信号
y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
      (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);

% 傅里叶变换
n = length(t);
f_omiga = fft(y_s);
omiga = (0:n-1) * (1/interval) / n;
f_omiga_shift = fftshift(f_omiga);
omiga_shift = (-n/2:n/2-1) * ((1/interval)/n);
result = abs(f_omiga_shift) / max(abs(f_omiga_shift));

% 计算频率分量占比
sumx = (sum(result) - 1) / 2;  % 所有频率分量求和(减去直流分量)
% sumx = result(round(n/2+1));

% 创建频率分量结构体
freq_components.sig_fund = result(round(n/2+f1 * n * interval) +1) / sumx;    % 主频分量 (f1)
freq_components.sig_harm = result(round(n/2+f2 * n * interval) +1) / sumx;    % 谐频分量 (f2)
freq_components.sig_sum = result(round((n/2+f1+f2) * n * interval) +1) / sumx;  % 合频分量 (f1+f2)
freq_components.sig_diff = result(round(n/2+abs(f2-f1) * n * interval) +1) / sumx; % 差频分量 (|f2-f1|)
freq_components.sig_double = result(round(n/2+2*f1 * n * interval)+1 ) / sumx;  % 二倍频分量 (2*f1)
freq_components.sig_triple = result(round(n/2+3*f1 * n * interval)+1 ) / sumx;  % 三倍频分量 (3*f1)

% 绘制频谱图
figure('Position', [100, 100, 1500, 500]);
plot(omiga_shift, result, 'linewidth', 2);
axis([-40001 40001 0 1]);
set(gca, 'Linewidth', 3, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Frequency (Hz)','FontWeight','bold','FontSize',24);
% set(gca, 'Linewidth', 3, 'FontWeight', 'bold', 'FontSize', 30);
title(sprintf('fmSAC Modulation Spectrum (f_1=%.1f KHz; f_2=%.1f KHz)', ...
    f1/1000, f2/1000),'FontWeight','bold','FontSize',24);

% 显示频率分量结果
disp('Frequency Components Analysis:');
disp(freq_components);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%测试FreqSAC_analysis.m函数方法如下：
clear
clc
close all

% 使用默认参数
% freq_comp = FreqSAC_analy();

% 使用自定义参数
freq_comp = FreqSAC_analy(...
    1.1e6, 0.49e6, 2.56e8, 2.7e-16, 10e3, 1000e3, ...
    532e-7, 561e-7, 15e3, 10e3, 1, 0.5, 1, 0.5e-6);