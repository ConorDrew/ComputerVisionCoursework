image_paths = train_image_paths;
vocab_size = 500;

clearvars -except image_paths vocab_size train_image_paths

% Var Setup
numImages = size(image_paths, 1); % number of training images
representativeData = 15; % Gets representative data from the sift feature.
% Can add these in to get sample data from further in the image.
repDataStart = 50;
repDataEnd = repDataStart + representativeData - 1;

stepSize = 3; % Size of the step in pixles over the image
binSize = 2; %A spatial bin covers SIZE pixels.

% Makes space for sift features to be stored.
%X = zeros(128 + 256, numImages * representativeData);
fprintf('Building vocab...\n');
fprintf('Settings:\n Representative Data: %d\n Step Size: %d \n', ...
    representativeData, stepSize);
fprintf('Representative Data range: %d - %d\n', ...
    repDataStart, repDataEnd);
counter = 0;
i = 1;

% for i=1:numImages
%     if (counter==50)
%         fprintf('%5.2f%% Complete.\n', (i/numImages)*100);
%         counter = 0;
%     end
    %Var setups for loop
    % when adding to X this will set what points depending on someSiftFeat
    descriptorStart = representativeData * (i-1) + 1;
    descriptorEnd = representativeData * i;
    
    % Gets image ready for sift.
    img = im2single(imread(image_paths{i}));
    %img = rgb2gray(img); % Not needed if using PHOW
    
    
    % Gets features and stores.
    %[~, features] = vl_dsift(img, 'Fast', 'Step', stepSize);
    [~, features] = vl_phow(img, 'Sizes', binSize, 'Fast', 'True', 'Step', stepSize, 'Color', 'rgb');

    X(:, descriptorStart : descriptorEnd) = features(:,1:representativeData);
    X2(:, descriptorStart : descriptorEnd) = features(:,repDataStart:repDataEnd);
    
    counter = counter + 1;
% end
   
% http://www.vlfeat.org/matlab/vl_kmeans.html for diffrent input
% values for Kmeans for testing.
% gets the centroids from K-Means
[centers, ~] = vl_kmeans(X, vocab_size);
vocab = single(centers');
    
%     %%
% close all
% input_images = train_image_paths;
% imgTemp = imread(char(input_images(1)));
% %imgTemp = rgb2hsv(imgTemp);
% 
% %imgTemp = rgb2hsv(imgTemp);
% Red = imgTemp(:,:,1);
% Green = imgTemp(:,:,2);
% Blue = imgTemp(:,:,3);
% 
% 
% % RGB color space
% [binR,edges] = imhist(Red, 16);
% [binG,edges] = imhist(Green, 16);
% [binB,edges] = imhist(Blue, 16);
% 
% %colorHist = binR' + binG' + binB';
% colorHist = cat(2, binR', binG', binB');

