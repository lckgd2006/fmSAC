# fmSAC: Frequency-Modulation Saturated Absorption Competition Microscopy

## Project Overview

This project focuses on the **nonlinear dynamics** of frequency-modulation saturated absorption competition microscopy (fmSAC). Through a series of MATLAB scripts, it systematically investigates core issues including the physical mechanism, parameter influences, imaging optimization, photobleaching performance, and 3D imaging capability of fmSAC. The work provides theoretical and data support for the practical application of fmSAC microscopy technology.

## Code File Descriptions

|Category|File Name|Description|
|---|---|---|
|Nonlinear Physical Mechanism Analysis|`freqSAC_nonlinear_v3.m`|Core file that conducts systematic discussion and derivation on the nonlinear physical mechanism of fmSAC technology, serving as the foundation for understanding fmSAC imaging principles.|
|Single-Point fmSAC Result Analysis and Parameter Exploration|`freqSAC_analysis_v3.m`|Performs basic analysis on the frequency-domain modulation saturated absorption competition (fmSAC) experimental/simulation results of a single detection point, outputting core characteristic data at the single-point level.|
||`freqSAC_Impact_v3.m`|Explores the influence of multiple key parameters (e.g., light intensity, modulation frequency, molecular concentration) on single-point fmSAC results, quantifying the correlation between parameter changes and imaging effects.|
||`freqSAC_Impact_lambda_v6.m`|Specializes in in-depth analysis of wavelength parameters, supporting two plotting modes (`combined` for comparative analysis and `individual` for single-wavelength independent analysis) to intuitively demonstrate the effect of different wavelengths on single-point fmSAC results.|
||`freqSAC_alpha_v4.m`|Proposes improvement schemes for fmSAC imaging quality by analyzing and optimizing the difference coefficient (α), enhancing imaging resolution and contrast.|
|Modulation Contrast and Imaging Optimization|`freqSAC_ContrastModulation_1_v2.m`|Investigates the excitation light modulation contrast, analyzes its influence mechanism on the overall fmSAC imaging effect, and clarifies the correlation between contrast and imaging quality.|
||`freqSAC_plus_cm1_v2.m`|Based on the analysis results of `freqSAC_ContrastModulation_1_v2`, optimizes fmSAC imaging performance by reducing the excitation light modulation contrast.|
||`freqSAC_ContrastModulation_2_v4.m`|Further explores the role of modulation contrast (focusing on the competition light dimension) in fmSAC imaging, supplementing analytical perspectives from different optical field modulation dimensions.|
||`freqSAC_plus_cm2_v7.m`|Specifically increases the competition light modulation contrast to verify its optimization effect on fmSAC imaging resolution and signal-to-noise ratio.|
|Photobleaching Performance Analysis|`freqSAC_bleaching_v7.m`|Compares the photobleaching characteristics of fmSAC imaging with traditional saturated absorption competition (SAC) microscopy, quantifying the performance advantages of fmSAC in reducing fluorophore photobleaching and extending imaging duration.|
||`freqSAC_bleaching_v6.m`|Builds on `freqSAC_bleaching_v7` to develop a visual GUI interface, supporting the viewing of imaging conditions at any scan number for interactive analysis of photobleaching changes during the scanning process.|
|3D Imaging Analysis|`freqSAC_imaging_3D_v6.m`|Simulates and analyzes the imaging capability of fmSAC for 3D randomly distributed fluorophores, verifying the feasibility and accuracy of fmSAC technology in 3D biological sample imaging.|
## Data File Descriptions

The project relies on the following pre-generated illumination light field data files (MATLAB .mat format) to simulate light field distributions under different wavelengths and illumination modes:

|File Name|Description|
|---|---|
|`I_exc532_51_3D.mat`|51×51×51 3D light field intensity data under 532nm wavelength and Gaussian illumination mode.|
|`I_exc488_51_3D.mat`|51×51×51 3D light field intensity data under 488nm wavelength and doughnut illumination mode.|
|`I_exc532_501.mat`|501×501 2D array light field intensity data under 532nm wavelength and Gaussian illumination mode.|
|`I_exc488_501.mat`|501×501 2D array light field intensity data under 488nm wavelength and doughnut illumination mode.|
## Auxiliary Function

- `save_subplots_separately.m`: A general-purpose subfunction used to independently save multiple subplots in MATLAB figures as image files (e.g., .png/.eps), facilitating the separate use of subplots in papers/reports.

## Recommended Code Execution Order

1. First run `freqSAC_nonlinear_v3.m` to understand the nonlinear mechanism of fmSAC, laying a theoretical foundation for subsequent analysis.

2. For single-point analysis, prioritize running `freqSAC_analysis_v3.m`, then explore parameter influences via `freqSAC_Impact_v3.m`/`freqSAC_Impact_lambda_v6.m`.

3. Imaging optimization codes (e.g., `freqSAC_plus_cm1_v2.m`/`freqSAC_plus_cm2_v7.m`) require prior execution of the corresponding contrast analysis codes (e.g., `freqSAC_ContrastModulation_1_v2.m`).

4. For 3D imaging analysis, ensure that data files such as `I_exc532_51_3D.mat`/`I_exc488_51_3D.mat` are in the same path as the code files.

5. The photobleaching GUI analysis (`freqSAC_bleaching_v6.m`) can be run directly, with imaging results viewed by selecting the scan number through the interface.
