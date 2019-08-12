% Based on James Hays, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_vocabulary( image_paths, vocab_size, stepSize,...
    binSize, colorSpace, representativeData, repDataStart, feature)
% The inputs are images, a N x 1 cell array of image paths and the size of 
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in make_hist.m. 

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.



% Var Setup
numImages = size(image_paths, 1); % number of training images
%representativeData = 15; % Gets representative data from the sift feature.
% Can add these in to get sample data from further in the image.
%repDataStart = 1;
repDataEnd = repDataStart + representativeData - 1;

%stepSize = 3; % Size of the step in pixles over the image
%binSize = 2; %A spatial bin covers SIZE pixels.
%colorSpace = 'rgb';

% Makes space for sift features to be stored.
switch lower(feature)
        case 'sift'
            if strcmp(colorSpace,'gray')
                X = zeros(128, numImages * representativeData);
            else
                X = zeros(128 + 256, numImages * representativeData);
            end
        case 'hog'
            if strcmp(colorSpace,'gray')
                X = zeros(31, numImages * representativeData);
            else
                X = zeros(31+48, numImages * representativeData);
                colourHist = get_color_histogram(image_paths, 16, colorSpace);  
            end
            
end


% if using colour add 256

fprintf('Building vocab...\n');
fprintf('Settings:\n Representative Data: %d\n Step Size: %d \n bin Size: %d\n', ...
    representativeData, stepSize, binSize);
fprintf('Representative Data range: %d - %d\n', ...
    repDataStart, repDataEnd);

counter = 0;
for i=1:numImages
    if (counter==100)
        fprintf('%5.2f%% Complete.\n', (i/numImages)*100);
        counter = 0;
    end
    features = [];
    %Var setups for loop
    % when adding to X this will set what points depending on someSiftFeat
    descriptorStart = representativeData * (i-1) + 1;
    descriptorEnd = representativeData * i;
    
    % Gets image ready for sift.
    img = im2single(imread(image_paths{i}));
    %img = rgb2gray(img); % Not needed if using PHOW
    
    
    % Gets features and stores.
    switch lower(feature)
        case 'sift'
            [~, features] = vl_phow(img, 'Sizes', binSize, 'Fast', 'True', 'Step', stepSize, 'Color', colorSpace);

        case 'hog'
            hog = vl_hog(img, stepSize);
            features = reshape(hog, 31, []);     
    end
    
    switch lower(feature)
        case 'sift'
            X(:, descriptorStart : descriptorEnd) = features(:,repDataStart:repDataEnd);
        case 'hog'
            if strcmp(colorSpace,'gray')
                X(:, descriptorStart : descriptorEnd) = features(:,repDataStart:repDataEnd);
            else
                C1 = repmat(colourHist(i,:)',1,representativeData);
                X(:, descriptorStart : descriptorEnd) = [features(:,repDataStart:repDataEnd);C1];
            end 
    end
    
    
    
    
    
    
    counter = counter + 1;
end
   
% http://www.vlfeat.org/matlab/vl_kmeans.html for diffrent input
% values for Kmeans for testing.
% gets the centroids from K-Means
fprintf('Finding K Means.\n');
[centers, ~] = vl_kmeans(X, vocab_size);
vocab = single(centers');

end
