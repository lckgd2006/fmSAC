%% fmSAC PSF Simulation/Molecular Imaging/Microtubule Imaging - 3D Version (Supports Slice Calculation)
%
% Includes 3D PSF and imaging simulation for Confocal, SAC, and fmSAC
% Set use_3d to false for 3D slicing, slice_z can be set from 1 to 51; in the "% Molecular Imaging Simulation" section, select 'random_2d' or 'microtubule_2d'
% Set use_3d to true for 3D volume calculation; in the "Molecular Imaging Simulation" section, select 'random_3d' or 'microtubule_3d'
%
clc; clear; close all;
addpath(genpath('PSF'));
tic;

%% Parameter Initialization
fprintf('Initializing parameters...\n');
params = struct(...
    'k_isc', 1.1e6, ...
    'k_t', 0.49e6, ...
    'k0', 2.56e8, ...
    'c1', 1 + 1.1e6/0.49e6, ...           % Pre-calculation
    'h', 6.626e-34, ...
    'c', 3e10, ...
    'lambda_s', 532e-7, ...
    'lambda_d', 488e-7, ...
    'sigma_s', 2.7e-16, ...
    'sigma_d', 2.7e-16 * 0.512063188, ... % Pre-calculation
    'I_s', 10e3, ...
    'I_d', 500e3, ...
    'f1', 10e3, ...
    'f2', 15e3, ...
    'interval', 10e-6, ...
    't', 0:10e-6:1-10e-6, ...             % Pre-calculated time series
    'm_s', 0.1, ...
    'm_d', 1.0, ...
    'use_3d', true, ...         % Set to true for 3D calculation, false for 2D slice calculation
    'slice_z', 1, ...           % Z-axis slice index when use_3d is false; slice_z = 25 or 26 is the focal plane slice
    'SidelobeCoeff',8 ...       % Parameter to control fmSAC sidelobe removal, recommended value is 8 for 51×51×51 PSF
);

%% Load 3D PSF Data
fprintf('Loading 3D PSF data...\n');
try
    % Replace with your actual 3D PSF data file
    exc_data = load('I_exc532_51_3D.mat');     
    hexc_data = load('I_hexc488_51_3D.mat'); 
  
    % Extract PSF data, adjust field names according to your data structure
    I_exc_psf = exc_data.result.PSF; 
    I_hexc_psf = hexc_data.result.PSF; 
    
    fprintf('PSF data loaded successfully\n');
    fprintf('Excitation PSF size: %s\n', mat2str(size(I_exc_psf)));
    fprintf('Competition PSF size: %s\n', mat2str(size(I_hexc_psf)));
    
catch ME
    fprintf('Failed to load PSF data: %s\n', ME.message);
    fprintf('Creating sample 3D PSF data...\n');
    
    % Create sample 3D PSF data (Gaussian distribution)
    [X, Y, Z] = meshgrid(-25:25, -25:25, -25:25);
   
    I_exc_psf = exp(-(X.^2 + Y.^2 + Z.^2)/(2*10^2));   % aussian spot
    theta = atan2(Y, X);                               % Angular coordinate
    I_hexc_psf = exp(-(X.^2 + Y.^2)/(2*8^2)) .* (X.^2 + Y.^2) .* exp(1i*theta) .* exp(-Z.^2/(2*6^2)); 
    % Vortex spot with topological charge
    I_hexc_psf = abs(I_hexc_psf);                      % Take intensity
end

% Normalization and scaling
I1 = I_exc_psf / max(I_exc_psf(:));
I2 = I_hexc_psf / max(I_hexc_psf(:));
I_exc = params.I_s * I1;
I_hexc = params.I_d * I2;

% Get data size
% if params.use_3d
[LL, MM, NN] = size(I_exc);
fprintf('3D PSF size: %d x %d x %d\n', LL, MM, NN);

%% Pre-calculate Constants (Improve Efficiency)
fprintf('Pre-calculating constants...\n');
const_s = params.sigma_s * params.lambda_s / (params.h * params.c);
const_d = params.sigma_d * params.lambda_d / (params.h * params.c);

%% Initialize Progress Bar
fprintf('Starting calculation of conventional SAC PSF...\n');
hWaitbar = waitbar(0, 'Calculating conventional SAC PSF...', 'Name', 'fmSAC Simulation Progress');

%% Calculate Conventional SAC PSF
% if params.use_3d
% 3D Calculation - GPU Acceleration
I_exc_gpu = gpuArray(I_exc);
I_hexc_gpu = gpuArray(I_hexc);
    
k_s = const_s * I_exc_gpu;
k_d = const_d * I_hexc_gpu;
y_SAC_gpu = k_s ./ (params.c1 * k_s + params.c1 * k_d + params.k0);
y_SAC = gather(y_SAC_gpu);
    
y_SAC = y_SAC / max(y_SAC(:));
    
waitbar(1, hWaitbar, 'Conventional SAC PSF calculation completed');

%% Calculate fmSAC
fprintf('\nCalculating fmSAC PSF...\n');

% Pre-calculate frequency indices 
n_time = length(params.t);
freq_res = (1/params.interval)/n_time;
f1_idx = round(params.f1/freq_res) + n_time/2 + 1;
f2_idx = round(params.f2/freq_res) + n_time/2 + 1;

waitbar(0.5, hWaitbar, sprintf('Calculating fmSAC @ %dkW/cm²...', params.I_d/1e3));

% if params.use_3d
% 3D Calculation - GPU Acceleration
I_exc_gpu = gpuArray(I_exc);
I_hexc_gpu = gpuArray(I_hexc);
    
% Pre-calculate modulation signals
cos_f1 = cos(2*pi*params.f1*params.t);
cos_f2 = cos(2*pi*params.f2*params.t);
    
% Initialize GPU arrays
sig_fund_gpu = gpuArray.zeros(LL, MM, NN);
sig_harm_gpu = gpuArray.zeros(LL, MM, NN);
    
% Calculate point by point
for m = 1:LL
    for n = 1:MM
        for p = 1:NN
            k_s = const_s * I_exc_gpu(m, n, p);
            k_d = const_d * I_hexc_gpu(m, n, p);
                
            % Generate modulation signal
            numerator = k_s * (1 + params.m_s * cos_f1);
            denominator = params.c1*(k_s*(1 + params.m_s * cos_f1) + ...
                k_d*(1 + params.m_d * cos_f2)) + params.k0;
            y_s = numerator ./ denominator;
                    
            % Spectrum analysis
            f_fft = fft(y_s);
            f_fft_shift = fftshift(f_fft);
            result = abs(f_fft_shift) / max(abs(f_fft_shift));
                    
            sumx = (sum(result) - result(n_time/2+1)) / 2;
            sig_fund_gpu(m, n, p) = result(f1_idx) / sumx;
            sig_harm_gpu(m, n, p) = result(f2_idx) / sumx;
        end
    end
        
    % Update progress
    progress = 0.5 + (m/LL)*0.4;
    waitbar(progress, hWaitbar, sprintf('Calculating fmSAC: %.1f%%', m/LL*100));
end
    
% Transfer data back to CPU
sig_fund = gather(sig_fund_gpu);
sig_harm = gather(sig_harm_gpu);
    
% Calculate fmSAC and remove sidelobes
alpha_val = min(sig_fund(:) ./ sig_harm(:));
fmSAC = sig_fund - alpha_val * sig_harm;
    
% Create mask to remove sidelobes
[x, y, z] = meshgrid(1:MM, 1:LL, 1:NN);
center = [ceil(LL/2), ceil(MM/2), ceil(NN/2)];
radius = sqrt((x - center(2)).^2 + (y - center(1)).^2 + (z - center(3)).^2);
fmSAC(radius > params.SidelobeCoeff) = 0;

%% Display PSF Images
waitbar(0.95, hWaitbar, 'Displaying PSF images...');

figure('Position', [50, 50, 1200, 900], 'Color', 'w');

if params.use_3d
    % Central slice display and profile for 3D data
    center_slice_xy = ceil(NN/2); % Central slice of XY plane
    center_slice_xz = ceil(MM/2); % Central slice of XZ plane (fixed Y)
    center_slice_yz = ceil(LL/2); % Central slice of YZ plane (fixed X)
    
    % XY plane
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
    
    % XZ plane (fixed Y)
    subplot(3,4,5);
    imagesc(rot90(squeeze(I_exc(:,center_slice_xz,:)), -1)); axis square; colorbar; 
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

    % Display fmSAC and SAC
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
    % Full display for 2D data
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

%% Molecular Imaging Simulation
waitbar(0.98, hWaitbar, 'Performing molecular imaging simulation...');

% Set physical size parameters
pixel_size = 50;     % nm/pixel, typical super-resolution microscope pixel size
axial_step = 50;     % nm/layer, Z-axis step size

% Set sample parameters
sample_params = struct(...
    'num_molecules', 200, ...        % Number of random molecules
    'sim_num', 15, ...               % Number of microtubules
    'step_num', 50, ...              % Number of microtubule steps
    'forces', 0, ...                 % Applied force
    'KT', 4.1, ...                   % Thermodynamic parameter
    'A', 1000, ...                   % Persistence length
    'l', 1, ...                      % Step size
    'sigma', 0.7, ...                % Gaussian smoothing parameter
    'segment_size', 1.5, ...         % Microtubule segment size
    'pixel_size', pixel_size, ...    % Pixel size (nm)
    'axial_step', axial_step ...     % Axial step size (nm)
);

sample_type = 'random_3d';           % or 'random_3d' or 'microtubule_3d'
physical_size = [5, 5, 5];           % Physical size [μm]: 5μm x 5μm x 5μm
% Convert to pixel size
dimensions = round(physical_size .* [1000, 1000, 1000] ./ [pixel_size, pixel_size, axial_step]);
fprintf('3D sample physical size: %.1fμm x %.1fμm x %.1fμm\n', physical_size);
fprintf('Corresponding pixel size: %d x %d x %d pixels\n', dimensions);
% Generate sample
s = generate_sample(sample_type, dimensions, sample_params);

% Convolution imaging
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

% Display imaging results
if params.use_3d
    figure('Position', [100, 100, 1000, 800], 'Color', 'w');
    center_slice_xy = ceil(size(s, 3)/2); % XY中心切片
    center_slice_xz = ceil(size(s, 2)/2); % XZ中心切片（固定Y）
    center_slice_yz = ceil(size(s, 1)/2); % YZ中心切片（固定X）
    
    % XY plane
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
    
    % XZ plane
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
    
    % YZ plane
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
    % 3D display of molecular or microtubule sample
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

%% Export Four Stacks as hot Pseudocolor 24-bit RGB TIFF
fprintf('Exporting hot pseudocolor RGB TIFF...\n');

stacks = {s, Iconf_sample, y_SAC_sample, fmSAC_sample};
names  = {'s_rgb', 'Iconf_rgb', 'y_SAC_rgb', 'fmSAC_rgb'};
% sli_z = [10,20,30,40,50,60,70,80,90,100];

% Pre-generate hot colormap
hotMap = hot(256);          % 256×3  double 0-1

for k = 1:numel(stacks)
    img = stacks{k};                    % Extract matrix
    gname = names{k};

    % Normalize to 0-255 index
    img = single(img);
    img = img - min(img(:));
    img = img / max(img(:)) * 255;
    img = uint8(img);                   % 0-255 integer

    % Grayscale → RGB (hot)
    if ndims(img) == 3
        % ========= 3D: Convert to RGB frame by frame =========
        [h, w, slices] = size(img);
        % t = Tiff(fname, 'w');
        % for i = 1:length(sli_z)
        for i = 1:slices
            % fname = gname + "_" + sli_z(i) + ".tif";
            fname = gname + "_" + i + ".tif";
            t = Tiff(fname, 'w');
            % idx = img(:, :, sli_z(i));
            idx = img(:, :, i);
            idx(idx==0) = 1;                  % Avoid 0 index
            rgb = ind2rgb(idx, hotMap);       % 0-1 double
            rgb = im2uint8(rgb);              % 0-255 uint8

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
        % ========= 2D: Single frame =========
        idx = img;
        idx(idx==0) = 1;
        rgb = ind2rgb(idx, hotMap);
        imwrite(rgb, fname, 'Compression', 'none');
    end
    fprintf('Exported %s (hot RGB)\n', fname);
end

fprintf('All completed!\n');

%% Completion
close(hWaitbar);
fprintf('Simulation completed!\n');

%% Auxiliary Function - Generate 2D or 3D Fluorescent Molecular/Microtubule Distribution
function sample = generate_sample(sample_type, dimensions, params)
% Generate fluorescent sample (random molecules or microtubule structures)
% Input:
%   sample_type: 'random_2d', 'random_3d', 'microtubule_2d', 'microtubule_3d'
%   dimensions: [xsize, ysize] or [xsize, ysize, zsize]
%   params: Structure containing sample parameters
%
% Output:
%   sample: Generated sample matrix

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
            error('Unsupported sample type: %s', sample_type);
    end
end

function sample = generate_random_2d(dimensions, params)
    % Generate 2D random fluorescent molecules
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
    % Generate 3D random fluorescent molecules
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
    % Generate 2D microtubule structures
    xsize = dimensions(1); ysize = dimensions(2);
    sim_num = params.sim_num; step_num = params.step_num;
    
    % Generate 3D microtubules and extract 2D slice
    microtubule_3d = generate_microtubule_3d([xsize, ysize, 1], params);
    sample = microtubule_3d(:, :, 1);
    
    % Optional: Add 2D-specific processing
    sample = imgaussfilt(sample, params.sigma);
end

function sample = generate_microtubule_3d(dimensions, params)
    % Generate 3D microtubule structures (Optimized version)
    xsize = dimensions(1); ysize = dimensions(2); zsize = dimensions(3);
    sim_num = params.sim_num; step_num = params.step_num;
    
    % Call WLC model to generate microtubule trajectories
    [wlcseries] = WLCmicrotubules_optimized(params.forces, params.KT, params.A, params.l, step_num, sim_num);
    
    % Create 3D image
    sample = zeros(xsize, ysize, zsize);
    
    for i = 1:sim_num
        % Normalize coordinates to image size
        coords = squeeze(wlcseries(:,:,i));
        coords_normalized = normalize_coordinates(coords, [xsize, ysize, zsize]);
        
        % Draw trajectory points to image
        for j = 1:size(coords_normalized, 1)
            x = round(coords_normalized(j, 1));
            y = round(coords_normalized(j, 2));
            z = round(coords_normalized(j, 3));
            
            if x >= 1 && x <= xsize && y >= 1 && y <= ysize && z >= 1 && z <= zsize
                % Draw microtubule segment (increase thickness)
                sample = draw_microtubule_segment(sample, [x, y, z], params.segment_size);
            end
        end
    end
    
    % Gaussian smoothing
    if zsize > 1
        sample = imgaussfilt3(sample, params.sigma);
    else
        sample = imgaussfilt(sample, params.sigma);
    end
end

function coords_normalized = normalize_coordinates(coords, target_size)
    % Normalize coordinates to target size
    coords_normalized = coords - min(coords);
    coords_normalized = coords_normalized ./ max(coords_normalized(:));
    coords_normalized = coords_normalized .* (target_size - 1) + 1;
end

function img = draw_microtubule_segment(img, center, radius)
    % Draw microtubule segment (increase thickness)
    [x, y, z] = meshgrid(1:size(img,2), 1:size(img,1), 1:size(img,3));
    distances = sqrt((x - center(2)).^2 + (y - center(1)).^2 + (z - center(3)).^2);
    img(distances <= radius) = 1;
end

function [wlcseries] = WLCmicrotubules_optimized(forces, KT, A, l, steptot, sim_num)
    % Optimized WLC microtubule generation function
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
fprintf('Code execution time: %.4f 秒\n', elapsedTime);