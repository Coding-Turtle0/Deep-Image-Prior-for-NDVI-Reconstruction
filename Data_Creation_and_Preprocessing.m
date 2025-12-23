
clc;
clear;
close all;

% FILE PATH
filename = 'MOD13Q1.A2024001.h31v09.061.2024022131951.hdf';

% INSPECT FILE STRUCTURE (OPTIONAL, FOR DEBUG)
info = hdfinfo(filename);
sds = info.Vgroup.Vgroup.SDS;

disp('Available datasets:');
for i = 1:length(sds)
    fprintf('%2d : %s\n', i, sds(i).Name);
end

% READ NDVI 
ndvi_raw = hdfread(filename, '250m 16 days NDVI');

% Apply MODIS scale factor
ndvi = double(ndvi_raw) * 0.0001;

% Remove invalid values
ndvi(ndvi < -1 | ndvi > 1) = NaN;

% READ CLOUD MASK
% 0 = good, 1 = marginal, 2 = snow/ice, 3 = cloud
qa = hdfread(filename, '250m 16 days pixel reliability');

cloudMask = double(qa == 0);   % 1 = clear, 0 = cloud/bad

% FORWARD MODEL: CLOUD-CORRUPTED NDVI
ndvi_cloudy = ndvi .* cloudMask;

% VISUAL SANITY CHECK
figure('Color','w');

subplot(1,3,1)
imshow(ndvi,[])
title('Clean NDVI (Ground Truth)')
colorbar

subplot(1,3,2)
imshow(cloudMask,[])
title('Cloud Mask (MODIS QA)')
colorbar

subplot(1,3,3)
imshow(ndvi_cloudy,[])
title('Cloud-Corrupted NDVI')
colorbar

% SAVE OUTPUTS FOR DEEP IMAGE PRIOR
imwrite(mat2gray(ndvi), 'ndvi_gt.tif');
imwrite(cloudMask, 'cloud_mask.tif');
imwrite(mat2gray(ndvi_cloudy), 'ndvi_cloudy.tif');

disp('NDVI and cloud mask successfully extracted.');


