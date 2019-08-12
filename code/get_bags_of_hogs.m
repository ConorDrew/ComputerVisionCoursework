% Implementated according to the starter code prepared by James Hays, Brown University
% Michal Mackiewicz, UEA, March 2015

function image_feats = get_bags_of_hogs(image_paths, stepSize, colorSpace)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.


load('vocab.mat');

vocab_size = size(vocab, 1); % Gets dimension of vocab.mat
numImages = size(image_paths, 1); % Gets number of images from test images

%stepSize = 3; % this should be the same as build_vocab stepsize
%binSize = 2; % A spatial bin covers SIZE pixels.
%colorSpace = 'rgb';

image_feats = zeros(numImages, vocab_size);

fprintf('Building Bag of Sifts...\n');
fprintf('Settings:\n Step Size: %d \n bin Size: %d\n', stepSize);

if ~strcmp(colorSpace,'gray')
    colourHist = get_color_histogram(image_paths, 16, colorSpace);
end

counter = 0;
for i=1:numImages
    if (counter==100)
        fprintf('%5.2f%% Complete.\n', (i/numImages)*100);
        counter = 0;
    end
    
    % Get image ready for SIFT
    img = im2single(imread(image_paths{i}));
    %img = rgb2gray(img); %not needed if using PHOW
    
    hog = vl_hog(img, stepSize);
    features = reshape(hog, 31, []);
    if ~strcmp(colorSpace,'gray')
        features = [features; repmat(colourHist(i,:)', 1, size(features,2))];
    end
    % Finds what cluster the Sift feature belongs too using KNN
    [indices, ~] = knnsearch(vocab, features', 'K', 3);
    
    % Add features into a histogram bin 
    [image_feats(i,:), ~] = histcounts(indices, vocab_size);
    
    counter = counter + 1;
end
fprintf('Complete.\n');
end
