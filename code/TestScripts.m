image_paths = train_image_paths;

clearvars -except image_paths vocab_size train_image_paths

%%

stepSize = 10; % Size of the step in pixles over the image
color = 'gray';

img = im2single(imread(image_paths{1}));
imgGray = rgb2gray(img);
imshow(img);
%%

% Runs Gray DSift
[~, featuresDSift] = vl_dsift(imgGray, 'Fast', 'Step', stepSize);

%%
clc
binSize = 1;
% Runs Gray Dsift using Phow
%[~, featuresPhow] = vl_phow(img, 'Fast', 'Step', stepSize, 'Color', 'gray');
[~, featuresPhow] = vl_phow(img, 'Sizes', binSize, 'Fast', 'True', 'Step', stepSize, 'Color', color);

%%



X = zeros(128, numImages * representativeData);

