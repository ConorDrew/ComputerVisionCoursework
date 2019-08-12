% Jan 2016, Michal Mackiewicz, UEA
% This code has been adapted from the code 
% prepared by James Hays, Brown University
clear
clc
close all
%% Step 0: Set up parameters, vlfeat, category list, and image paths.

%FEATURE = 'tiny image';
%FEATURE = 'colour histogram';
FEATURE = 'bag of sift';
% FEATURE = 'bag of hog';
%FEATURE = 'spatial pyramids';

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

% Set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
%run('vlfeat/toolbox/vl_setup')
%run('C:\Users\izzyp\OneDrive\Documents\MATLAB\computer-vision-cw-2\vlfeat-0.9.21\toolbox\vl_setup')
data_path = '../data/';

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)

% Vars for build_vocab
    vocab_size = 500; % you need to test the influence of this parameter(clusters)
    representativeData = 15;
    repDataStart = 1;
    colorSpace = 'rgb'; % gray, rgb, hsv, opponent
    
%Vars for build and bag
    stepSize = 3;
    binSize = 2;


switch lower(FEATURE)    
    case 'tiny image'
        
        train_image_feats = get_tiny_images(train_image_paths, 16);
        test_image_feats  = get_tiny_images(test_image_paths, 16);
    case 'colour histogram'
        colorSpace = 'rgb';
        train_image_feats = get_color_histogram(train_image_paths,16, colorSpace);
        test_image_feats  = get_color_histogram(test_image_paths,16, colorSpace);
     case 'bag of sift'
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            
            % Vars for build_vocab
%             vocab_size = 500; % you need to test the influence of this parameter(clusters)
%             representativeData = 15;
%             repDataStart = 1;
            
            vocab = build_vocabulary(train_image_paths, vocab_size, ...
                stepSize, binSize, colorSpace, ...
                representativeData, repDataStart, 'sift'); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end
        
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats.mat', 'file')
            disp('Getting training bag of sifts');
            train_image_feats = get_bags_of_sifts(train_image_paths, ...
                stepSize, binSize, colorSpace); %Allow for different sift parameters
            disp('Getting testing bag of sifts');
            test_image_feats  = get_bags_of_sifts(test_image_paths, ...
                stepSize, binSize, colorSpace); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        else
            fprintf('Load training features:');
            load('image_feats.mat');
        end
      % YOU CODE spatial pyramids method
      case 'spatial pyramids'
        if exist('vocab.mat', 'file')
            load('vocab.mat');
            stepSize = 3;
            binSize = 2;
            colorSpace = 'rgb'; % gray, rgb, hsv, opponent
%             imgTest = imread("C:\Users\izzyp\OneDrive\Documents\MATLAB\computer-vision-cw-2\data\test\kitchen\sun_aahutlxpfsmsmmqi.jpg");
%             img = im2single(imgTest);
%             test = spatial_pyramids(img, stepSize, binSize, colorSpace, vocab);          
        end
        
    case 'bag of hog'
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')

            vocab = build_vocabulary(train_image_paths, vocab_size, ...
                stepSize, binSize, colorSpace, ...
                representativeData, repDataStart, 'hog'); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end
        
        if ~exist('image_feats.mat', 'file')
            disp('Getting training bag of sifts');
            train_image_feats = get_bags_of_hogs(train_image_paths, ...
                stepSize, colorSpace); %Allow for different sift parameters
            disp('Getting testing bag of sifts');
            test_image_feats  = get_bags_of_hogs(test_image_paths, ...
                stepSize, colorSpace); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        else
            fprintf('Load training features:');
            load('image_feats.mat');
        end

end
%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        predicted_categories = nearest_neighbor_classify(7, train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);
end

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
                    
                    
beep
pause(0.2)
beep
pause(0.2)
beep
pause(0.2)
beep
pause(0.2)