function freq_components = FreqSAC_analy(k_isc, ...
    k_t, k0, sigma_s, I_s, I_d, lambda_s, lambda_d, f1, f2, m_s, m_d, duration, interval)

%% FreqSAC_analysis: Analyze the spectral characteristics of dual-modulation SAC signals
%
% Input Parameters:
%   k_isc      - Intersystem crossing rate (default: 1.1e6)
%   k_t        - Triplet state decay rate (default: 0.49e6)
%   k0         - Fluorescence decay rate (default: 2.56e8)
%   sigma_s    - Excitation light absorption cross-section (default: 2.7e-16)
%   I_s        - Excitation light intensity (W/cm²) (default: 10e3)
%   I_d        - Competition light intensity (W/cm²) (default: 1000e3)
%   lambda_s   - Excitation light wavelength (cm) (default: 532e-7)
%   lambda_d   - Competition light wavelength (cm) (default: 561e-7)
%   f1         - Excitation modulation frequency (Hz) (default: 10e3)
%   f2         - Competition modulation frequency (Hz) (default: 15e3)
%   m_s        - Excitation modulation contrast (default: 1.0)
%   m_d        - Competition modulation contrast (default: 0.5)
%   duration   - Signal duration (s) (default: 1)
%   interval   - Sampling interval (s) (default: 0.5e-6)
%
% Output Parameters:
%   freq_components - Structure containing the proportion of each frequency component
%   omiga_shift     - Frequency axis data (Hz)
%   result          - Normalized spectral amplitude

%% Parameter Initialization
% Set default parameters
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

% Calculate relevant constants
h = 6.626e-34;      % Planck constant (J·s)
c = 3e10;           % Speed of light (cm/s)
c1 = 1 + k_isc/k_t; % Precomputed constant for rate equation

% Calculate competition light absorption cross-section (based on R6G dye ratio)
sigma_d = sigma_s * 0.049850201;

% Calculate excitation and depletion rates
k_s = sigma_s * I_s * lambda_s / (h * c);
k_d = sigma_d * I_d * lambda_d / (h * c);

%% Calculate fmSAC signal
% Generate time sequence
t = 0:interval:duration-interval;

% Generate modulated signal
y_s = (k_s * (1 + m_s * cos(2*pi*f1*t))) ./ ...
      (c1 * (k_s * (1 + m_s * cos(2*pi*f1*t)) + k_d * (1 + m_d * cos(2*pi*f2*t))) + k0);

% Fourier transform
n = length(t);
f_omiga = fft(y_s);
omiga = (0:n-1) * (1/interval) / n;
f_omiga_shift = fftshift(f_omiga);
omiga_shift = (-n/2:n/2-1) * ((1/interval)/n);
result = abs(f_omiga_shift) / max(abs(f_omiga_shift));

% Calculate the proportion of each frequency component
sumx = (sum(result) - 1) / 2;  % Sum of all frequency components (excluding DC component)
% sumx = result(round(n/2+1));

% Create structure to store frequency component proportions
freq_components.sig_fund = result(round(n/2+f1 * n * interval) +1) / sumx;         % Fundamental frequency component (f1)
freq_components.sig_harm = result(round(n/2+f2 * n * interval) +1) / sumx;         % Harmonic frequency component (f2)
freq_components.sig_sum = result(round((n/2+f1+f2) * n * interval) +1) / sumx;     % Sum frequency component (f1+f2)
freq_components.sig_diff = result(round(n/2+abs(f2-f1) * n * interval) +1) / sumx; % Difference frequency component (|f2-f1|)
freq_components.sig_double = result(round(n/2+2*f1 * n * interval)+1 ) / sumx;     % Second harmonic component (2*f1)
freq_components.sig_triple = result(round(n/2+3*f1 * n * interval)+1 ) / sumx;     % Third harmonic component (3*f1)

%% Visualization
% Plot spectrum
figure('Position', [100, 100, 1500, 500]);
plot(omiga_shift, result, 'linewidth', 2);
axis([-40001 40001 0 1]);
set(gca, 'Linewidth', 3, 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Normalized Intensity (a.u.)','FontWeight','bold','FontSize',24);
xlabel('Frequency (Hz)','FontWeight','bold','FontSize',24);
% set(gca, 'Linewidth', 3, 'FontWeight', 'bold', 'FontSize', 30);
title(sprintf('fmSAC Modulation Spectrum (f_1=%.1f KHz; f_2=%.1f KHz)', ...
    f1/1000, f2/1000),'FontWeight','bold','FontSize',24);

% Display frequency component analysis results
disp('Frequency Components Analysis:');
disp(freq_components);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test Method for FreqSAC_analysis.m:
clear
clc
close all

% Use default parameters
% freq_comp = FreqSAC_analy();

% Use custom parameters
freq_comp = FreqSAC_analy(...
    1.1e6, 0.49e6, 2.56e8, 2.7e-16, 10e3, 1000e3, ...
    532e-7, 561e-7, 15e3, 10e3, 1, 0.5, 1, 0.5e-6);