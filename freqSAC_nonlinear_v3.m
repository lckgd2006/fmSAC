%% %下面绘制实心斑荧光量与光强的曲线(二能级、三能级、四能级、五能级饱和)
%%考虑漂白因素，
%Eggeling C, "Molecular photobleaching kinetics..." Chemphyschem 6(5),791-804(2005).
%总单位时间s，长度单位cm
clc;clear;close all
lambda=532e-7;%波长cm
I=0:1e4:1e8;%单位W/cm2
Iav=I./2;
h=6.626e-34;%普兰克常数
c=3e10;%光速cm/s
phif=0.02;%荧光探测效率
tob=0.24e-3;  %重要，扫描的时候每点停留时间，单位是s
k0=2.56*10^8;%基态迁移速率，单位为s-1
kf=2.4*10^8;%S1态自发荧光速率
kisc=1.1*10^6;%S1到T1的迁移速率
kt=4.9e5;%T1到S0的迁移速率
kb=650;%S1与T1态的漂白速率
ksn1=5e12;%Sn到S1的迁移速率
ktn1=ksn1;
kbsn=2.8*10^8;%Sn态的漂白速率
kbtn=2.8*10^8;%Tn态的漂白速率
sig01=2.22e-16;%S0到S1的吸收截面
sig1n=0.77e-17;%S1到Sn的吸收截面
sigt1n=3.85e-17;%T1到Tn的吸收截面
PHIf=kf/k0;%发射量子效率
gamma=lambda/(h*c);
k01=sig01.*I*gamma;
k1n=sig1n.*I*gamma;
kt1n=sigt1n.*I*gamma;
kbn=kbtn/ktn1*(sig1n*gamma+kisc/kt*sigt1n*gamma);

pb=kb/k0+kbn/k0.*I;
x2=1*(1*(1*(k0+k01)+k01.*1))+(1+1)*1*1.*k01;
s0eq2=1*1*1*k0./x2;
s1eq2=s0eq2.*k01/k0;
out2=PHIf*phif.*s1eq2*k0*tob;%二能级饱和
pb2=1/k0+1/k0.*I;%kb,kbn均为1
kz2=(1+1.*I).*s1eq2;
out_b2=PHIf*phif./pb2.*(1-exp(-kz2*tob));%%二能级漂白情况下

x3=1*(kt*(1*(k0+k01)+k01.*1))+(1+1)*kisc*1.*k01;
s0eq3=1*1*kt*k0./x3;
s1eq3=s0eq3.*k01/k0;
out3=PHIf*phif.*s1eq3*k0*tob;%三能级饱和
pb3=kb/k0+1/k0.*I;%kbn为1
kz3=(kb+1.*I).*s1eq3;
out_b3=PHIf*phif./pb3.*(1-exp(-kz3*tob));%%三能级漂白情况下

x4=1*(kt*(ksn1*(k0+k01)+k01.*k1n))+(1+1)*kisc*ksn1.*k01;
s0eq4=1*ksn1*kt*k0./x4;
s1eq4=s0eq4.*k01/k0;
out4=PHIf*phif.*s1eq4*k0*tob;%四能级饱和
pb4=kb/k0+(kbn*0.5)/k0.*I;%kbn之所以乘以0.5，因为Tn能级没有，只取kbn的一半
kz4=(kb+kbn*0.5.*I).*s1eq4;
out_b4=PHIf*phif./pb4.*(1-exp(-kz4*tob));%%四能级漂白情况下

x5=ktn1*(kt*(ksn1*(k0+k01)+k01.*k1n))+(kt1n+ktn1)*kisc*ksn1.*k01;
s0eq5=ktn1*ksn1*kt*k0./x5;
s1eq5=s0eq5.*k01/k0;
out5=PHIf*phif.*s1eq5*k0*tob;%五能级饱和
pb5=kb/k0+kbn/k0.*I;
kz5=(kb+kbn.*I).*s1eq5;
out_b5=PHIf*phif./pb5.*(1-exp(-kz5*tob));%%五能级漂白情况下


% % 学术经典版 - 使用MATLAB经典colormap
% figure('Position', [100, 100, 1000, 600])
% colors = parula(5); % 使用parula colormap生成6种颜色
% 
% % 绘制饱和曲线 (实线)
% p2 = semilogx(I, out3, '--', 'Color', colors(2,:), 'LineWidth', 2.5);
% p3 = semilogx(I, out4, '-',  'Color', colors(3,:), 'LineWidth', 2.5);
% p4 = semilogx(I, out5, '-',  'Color', colors(4,:), 'LineWidth', 2.5);
% 
% % 绘制漂白曲线 (虚线)
% p5 = semilogx(I, out_b4, ':', 'Color', colors(5,:), 'LineWidth', 3);
% p6 = semilogx(I, out_b5, ':', 'Color', colors(6,:), 'LineWidth', 3);
% 
% % 图形美化
% xlabel('Excitation Intensity (W/cm^2)', 'FontSize', 13, 'FontWeight', 'bold');
% ylabel('Fluorescence Intensity (A.U.)', 'FontSize', 13, 'FontWeight', 'bold');
% title('Nonlinear Saturation and Bleaching Dynamics', 'FontSize', 14, 'FontWeight', 'bold');
% legend([p2, p3, p4, p5, p6], ...
%        {'3-level Saturated', '4-level Saturated', '5-level Saturated', ...
%         '4-level Bleaching', '5-level Bleaching'}, ...
%        'Location', 'northeast', 'FontSize', 11);
% grid on; set(gca, 'FontSize', 12, 'LineWidth', 1.2);


% 学术经典版（完整8曲线）- 使用MATLAB经典colormap
figure('Position', [100, 100, 1000, 600])
colors = parula(6); % 使用parula colormap生成8种颜色

% 绘制所有饱和曲线 (实线)
p1 = semilogx(I, out2, '-', 'Color', colors(2,:), 'LineWidth', 2.5); hold on;
p2 = semilogx(I, out3, '-', 'Color', colors(3,:), 'LineWidth', 2.5);
p3 = semilogx(I, out4, '-', 'Color', colors(4,:), 'LineWidth', 2.5);
p4 = semilogx(I, out5, '-', 'Color', colors(5,:), 'LineWidth', 2.5);

% 绘制所有漂白曲线 (虚线)
p5 = semilogx(I, out_b2, '--', 'Color', colors(2,:), 'LineWidth', 2.5);
p6 = semilogx(I, out_b3, '--', 'Color', colors(3,:), 'LineWidth', 2.5);
p7 = semilogx(I, out_b4, '--', 'Color', colors(4,:), 'LineWidth', 2.5);
p8 = semilogx(I, out_b5, '--', 'Color', colors(5,:), 'LineWidth', 2.5);

% 图形美化
set(gca,'FontSize',18,'FontWeight','bold','LineWidth',2); 
xlabel('Excitation Intensity (W/cm^2)', 'FontSize',24,'FontWeight','bold');
ylabel('Fluorescence Intensity (a.u.)', 'FontSize',24,'FontWeight','bold');
title('Nonlinear Saturation and Bleaching Dynamics in Multi-level Systems','FontSize',24,'FontWeight','bold');

% 创建图例 - 按顺序排列
legend([p1, p2, p3, p4, p5, p6, p7, p8],...
       {'2-level Saturation', '3-level Saturation', '4-level Saturation', '5-level Saturation', ...
        '2-level + Bleaching', '3-level + Bleaching', '4-level + Bleaching', '5-level + Bleaching'}, ...
       'Location', 'northwest', 'FontSize', 12, 'NumColumns', 2); % 两列布局
legend('boxoff');
grid on; 
% set(gca, 'FontSize', 12, 'LineWidth', 1.2);

% 添加网格和边框美化
set(gca, 'GridAlpha', 0.3, 'GridColor', [0.3 0.3 0.3]);
box on;