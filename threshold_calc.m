clear all; close all; clc
%% include all the adc data .xlsx files into the same folder as this matlab code
%% 
dataD10H10 = readmatrix('adc_10h10d.xlsx');
dataD10H30 = readmatrix('adc_30h10d.xlsx');
dataD30H10 = readmatrix('adc_10h30d.xlsx');
dataD30H30 = readmatrix('adc_30h30d.xlsx');

rmsD10H10 = dataD10H10(:,3);
rmsD10H30 = dataD10H30(:,3);
rmsD30H10 = dataD30H10(:,3);
rmsD30H30 = dataD30H30(:,3);

%% Normalised RMS Threshold, Peak Threshold, RMS Threshold
peakD10H10 = dataD10H10(:,2);
peakD10H30 = dataD10H30(:,2);
ratioD10H10 = rmsD10H10./peakD10H10;
ratioD10H30 = rmsD10H30./peakD10H30;

norm_rmsD10H10 = mean(ratioD10H10);
norm_rmsD10H30 = mean(ratioD10H30);
norm_rms_thres = (norm_rmsD10H30+norm_rmsD10H10)/2

mean_peakD10H10 = mean(peakD10H10);
mean_peakD10H30 = mean(peakD10H30);
peak_thres = (mean_peakD10H30+mean_peakD10H10)/2

mean_rmsD30H10 = mean(rmsD30H10);
mean_rmsD30H30 = mean(rmsD30H30);
rms_thres = (mean_rmsD30H10+mean_rmsD30H30)/2

%% Decay Threshold
decayD30H10 = dataD30H10(:,7);
decayD30H30 = dataD30H30(:,7);
avg_decayD30H10 = mean(decayD30H10);
avg_decayD30H30 = mean(decayD30H30);
decay_thres = (avg_decayD30H30+avg_decayD30H10)/2

%% Centroid Threshold
% centroidD10 = [dataD10H10(:, 6); dataD10H30(:, 6)];
% centroidD30 = [dataD30H10(:, 6); dataD30H30(:, 6)];
% avg_centroidD10 = mean(centroidD10);
% avg_centroidD30 = mean(centroidD30);
% centroid_thres = (avg_centroidD10 + avg_centroidD30) / 2