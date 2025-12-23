
clc;
clear;
close all;


% GPU CHECK
g = gpuDevice;
fprintf('Using GPU: %s\n', g.Name);


% LOAD DATA
ndvi_gt      = im2double(imread('ndvi_gt.tif'));
ndvi_cloudy = im2double(imread('ndvi_cloudy.tif'));
cloudMask   = im2double(imread('cloud_mask.tif'));

ndvi_gt      = ndvi_gt(:,:,1);
ndvi_cloudy = ndvi_cloudy(:,:,1);
cloudMask   = cloudMask(:,:,1);

% EXTRACT PATCH 
patchSize = 256;
rowStart  = 400;
colStart  = 600;

ndvi_gt_p      = ndvi_gt(rowStart:rowStart+patchSize-1, ...
                         colStart:colStart+patchSize-1);

ndvi_cloudy_p = ndvi_cloudy(rowStart:rowStart+patchSize-1, ...
                            colStart:colStart+patchSize-1);

cloudMask_p   = cloudMask(rowStart:rowStart+patchSize-1, ...
                          colStart:colStart+patchSize-1);

[H, W] = size(ndvi_gt_p);
fprintf('Running DIP on patch: %d x %d\n', H, W);

% WATER MASK
% NDVI < 0 usually corresponds to water or non-vegetation
% We EXCLUDE these pixels from the loss and metrics
waterMask = ndvi_gt_p > 0;   % 1 = vegetation, 0 = water

% Combined valid mask:
% - not cloud
% - not water
validMask = cloudMask_p .* waterMask;


% FIXED RANDOM INPUT (GPU)
inputDepth = 32;
z = randn(H, W, inputDepth, 'single', 'gpuArray');
dlZ = dlarray(z, 'SSCB');

% DIP NETWORK 
layers = [
    imageInputLayer([H W inputDepth],'Normalization','none')

    convolution2dLayer(3,64,'Padding','same')
    leakyReluLayer(0.1)

    convolution2dLayer(3,64,'Padding','same')
    leakyReluLayer(0.1)

    convolution2dLayer(3,1,'Padding','same')
];

dlnet = dlnetwork(layerGraph(layers));


% MOVE DATA TO GPU
y_dl     = dlarray(gpuArray(single(ndvi_cloudy_p)),'SSCB');
mask_dl  = dlarray(gpuArray(single(validMask)),'SSCB');


% OPTIMIZATION SETTINGS
numIter   = 2000;
learnRate = 1e-3;
noiseStd  = 0.03;     % canonical DIP noise injection

trailingAvg = [];
trailingAvgSq = [];

lossHist = zeros(numIter,1);

% DIP OPTIMIZATION LOOP (GPU + EARLY STOPPING)
figure('Name','DIP Optimization','Color','w');

for iter = 1:numIter

    % Input noise injection (VERY IMPORTANT) ----
    dlZ_noisy = dlZ + noiseStd * randn(size(dlZ),'like',dlZ);

    [loss, grads, xhat] = dlfeval(@dipLoss, dlnet, dlZ_noisy, y_dl, mask_dl);

    [dlnet, trailingAvg, trailingAvgSq] = adamupdate( ...
        dlnet, grads, trailingAvg, trailingAvgSq, iter, learnRate);

    lossHist(iter) = extractdata(loss);

    % Early stopping (KEY FOR DIP) 
    if iter > 600 && lossHist(iter) > lossHist(iter-200)
        disp('Early stopping triggered');
        break;
    end

    % Visualization
    if mod(iter,200) == 0
        ndvi_tmp = gather(extractdata(xhat));
        ndvi_tmp = max(min(ndvi_tmp,1),0);

        subplot(1,3,1)
        imshow(ndvi_cloudy_p,[])
        title('Clouded NDVI')

        subplot(1,3,2)
        imshow(ndvi_tmp,[])
        title(['DIP Reconstruction | Iter ',num2str(iter)])

        subplot(1,3,3)
        plot(lossHist(1:iter),'LineWidth',1.5)
        title('Masked MSE Loss')
        xlabel('Iteration'); ylabel('Loss')

        drawnow;
    end
end

% FINAL RECONSTRUCTION
ndvi_rec = gather(extractdata(xhat));
ndvi_rec = max(min(ndvi_rec,1),0);

% QUANTITATIVE EVALUATION (WATER-EXCLUDED, TYPE-SAFE)
evalMask = validMask == 1;

A   = double(ndvi_rec)   .* double(evalMask);
REF = double(ndvi_gt_p) .* double(evalMask);

rmse = sqrt(mean((A(evalMask) - REF(evalMask)).^2));
psnr_val = psnr(A, REF);
ssim_val = ssim(A, REF);

fprintf('\nFINAL DIP RESULTS (VEGETATION ONLY)\n');
fprintf('RMSE  = %.4f\n', rmse);
fprintf('PSNR = %.2f dB\n', psnr_val);
fprintf('SSIM = %.4f\n', ssim_val);

% FINAL VISUALIZATION
figure('Name','Final Results','Color','w');

subplot(1,5,1), imshow(ndvi_gt_p,[]), title('GT NDVI')
subplot(1,5,2), imshow(ndvi_cloudy_p,[]), title('Clouded NDVI')
subplot(1,5,3), imshow(ndvi_rec,[]), title('DIP Reconstruction')
subplot(1,5,4), imshow(waterMask,[]), title('Water Mask')
subplot(1,5,5), imshow(abs(ndvi_rec - ndvi_gt_p),[]), title('Abs Error')
colorbar

% SAVE OUTPUT
save('ndvi_DIP_reconstruction_patch.mat', 'ndvi_rec');
disp(' FINAL GPU DIP reconstruction completed.');

% LOSS FUNCTION (MASKED MSE, WATER-AWARE)
function [loss, grads, xhat] = dipLoss(dlnet, dlZ, y, mask)

    xhat = forward(dlnet, dlZ);

    diff = (xhat - y) .* mask;
    loss = mean(diff.^2,'all');

    grads = dlgradient(loss, dlnet.Learnables);
end
