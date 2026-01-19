%% Plot the curve of fluorescent intensity of solid spots vs light intensity (two-level, three-level, four-level, five-level saturation)
%
% Consider the bleaching factor,Eggeling C, "Molecular photobleaching kinetics..." Chemphyschem 6(5),791-804(2005).
% otal time unit: s, length unit: cm
%
clc;clear;close all
lambda=532e-7;          % Wavelength (cm)
I=0:1e4:1e8;            % Excitation intensity (W/cm2)
Iav=I./2;
h=6.626e-34;            % Planck constant
c=3e10;                 % Speed of light (cm/s)
phif=0.02;              % Fluorescence detection efficiency
tob=0.24e-3;            % Dwell time per point during scanning (s)
k0=2.56*10^8;           % Ground state transition rate (s-1)
kf=2.4*10^8;            % Spontaneous fluorescence rate of S1 state
kisc=1.1*10^6;          % Transition rate from S1 to T1
kt=4.9e5;               % Transition rate from T1 to S0
kb=650;                 % Bleaching rate of S1 and T1 states
ksn1=5e12;              % Transition rate from Sn to S1
ktn1=ksn1;
kbsn=2.8*10^8;          % Bleaching rate of Sn state
kbtn=2.8*10^8;          % Bleaching rate of Tn state
sig01=2.22e-16;         % Absorption cross-section from S0 to S1
sig1n=0.77e-17;         % Absorption cross-section from S1 to Sn
sigt1n=3.85e-17;        % Absorption cross-section from T1 to Tn
PHIf=kf/k0;             % Emission quantum efficiency
gamma=lambda/(h*c);
k01=sig01.*I*gamma;
k1n=sig1n.*I*gamma;
kt1n=sigt1n.*I*gamma;
kbn=kbtn/ktn1*(sig1n*gamma+kisc/kt*sigt1n*gamma);
pb=kb/k0+kbn/k0.*I;

x2=1*(1*(1*(k0+k01)+k01.*1))+(1+1)*1*1.*k01;
s0eq2=1*1*1*k0./x2;
s1eq2=s0eq2.*k01/k0;
out2=PHIf*phif.*s1eq2*k0*tob;    % Two-level saturation
pb2=1/k0+1/k0.*I;                % kb and kbn are both set to 1
kz2=(1+1.*I).*s1eq2;
out_b2=PHIf*phif./pb2.*(1-exp(-kz2*tob));   % Two-level system with bleaching

x3=1*(kt*(1*(k0+k01)+k01.*1))+(1+1)*kisc*1.*k01;
s0eq3=1*1*kt*k0./x3;
s1eq3=s0eq3.*k01/k0;
out3=PHIf*phif.*s1eq3*k0*tob;    % Three-level saturation
pb3=kb/k0+1/k0.*I;%kbn为1
kz3=(kb+1.*I).*s1eq3;
out_b3=PHIf*phif./pb3.*(1-exp(-kz3*tob));   % Three-level system with bleaching

x4=1*(kt*(ksn1*(k0+k01)+k01.*k1n))+(1+1)*kisc*ksn1.*k01;
s0eq4=1*ksn1*kt*k0./x4;
s1eq4=s0eq4.*k01/k0;
out4=PHIf*phif.*s1eq4*k0*tob;    % Four-level saturation
pb4=kb/k0+(kbn*0.5)/k0.*I;       % kbn is multiplied by 0.5 because Tn level does not exist, only half of kbn is taken
kz4=(kb+kbn*0.5.*I).*s1eq4;
out_b4=PHIf*phif./pb4.*(1-exp(-kz4*tob));   % Four-level system with bleaching

x5=ktn1*(kt*(ksn1*(k0+k01)+k01.*k1n))+(kt1n+ktn1)*kisc*ksn1.*k01;
s0eq5=ktn1*ksn1*kt*k0./x5;
s1eq5=s0eq5.*k01/k0;
out5=PHIf*phif.*s1eq5*k0*tob;    % Five-level saturation
pb5=kb/k0+kbn/k0.*I;
kz5=(kb+kbn.*I).*s1eq5;
out_b5=PHIf*phif./pb5.*(1-exp(-kz5*tob));   % Five-level system with bleaching

% Academic classic version (complete 8 curves) - using MATLAB classic colormap
figure('Position', [100, 100, 1000, 600])
colors = parula(6); % Generate 8 colors using parula colormap

% Plot all saturation curves (solid lines)
p1 = semilogx(I, out2, '-', 'Color', colors(2,:), 'LineWidth', 2.5); hold on;
p2 = semilogx(I, out3, '-', 'Color', colors(3,:), 'LineWidth', 2.5);
p3 = semilogx(I, out4, '-', 'Color', colors(4,:), 'LineWidth', 2.5);
p4 = semilogx(I, out5, '-', 'Color', colors(5,:), 'LineWidth', 2.5);

% Plot all bleaching curves (dashed lines)
p7 = semilogx(I, out_b4, '--', 'Color', colors(4,:), 'LineWidth', 2.5);
p8 = semilogx(I, out_b5, '--', 'Color', colors(5,:), 'LineWidth', 2.5);

% Figure beautification
set(gca,'FontSize',18,'FontWeight','bold','LineWidth',2); 
xlabel('Excitation Intensity (W/cm^2)', 'FontSize',24,'FontWeight','bold');
ylabel('Fluorescence Intensity (a.u.)', 'FontSize',24,'FontWeight','bold');
title('Nonlinear Saturation and Bleaching Dynamics in Multi-level Systems','FontSize',24,'FontWeight','bold');

% Create legend - ordered arrangement
legend([p1, p2, p3, p4, p7, p8],...
       {'2-level Saturation', '3-level Saturation', '4-level Saturation', '5-level Saturation', ...
        '4-level + Bleaching', '5-level + Bleaching'}, ...
       'Location', 'northwest', 'FontSize', 12, 'NumColumns', 2);
legend('boxoff');
grid on; 
% set(gca, 'FontSize', 12, 'LineWidth', 1.2);

% Add grid and border beautification
set(gca, 'GridAlpha', 0.3, 'GridColor', [0.3 0.3 0.3]);
box on;