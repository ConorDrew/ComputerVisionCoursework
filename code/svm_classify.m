% Based on James Hays, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

cats = unique(train_labels, 'stable'); % gets labels for each categorie.
num_cats = length(cats); % Gets number of categories

num_train = size(train_image_feats, 1); % get number of training data

featSize = size(test_image_feats, 2); % gets size of feature to store
wStored = zeros(num_cats, featSize); % Stores the values of W 
bStored = zeros(num_cats, 1); % Store the values of B
labels = zeros(num_train,1); % Makes a list to store labels

LAMBDA = 10;
numIterations = 1000000;

for i=1:num_cats
    
    labels(:,1) = -1; % Stores all to -1
    
    % strCmp returns 1 if current cat index matches train label
    % labels then takes that input and converts the true sections to 1
    % Leaving the rest as -1.
    labels(strcmp(cats{i}, train_labels)) = 1;
    
    % Trains with SVM trian, with ablity to change LAMBDA and Iterations,
    [W, B] = vl_svmtrain(train_image_feats', labels, LAMBDA,...
                                    'MaxNumIterations', numIterations);
                                
    wStored(i,:) = W'; % Store transposed W at index
    bStored(i) = B; % Stores B at index
    
    
    

    
end
% stores the condfidence of if the image belongs in the cat
confidence = wStored*test_image_feats'; 
% Find the max confidence index and store that as the cat.
[~, indices] = max(confidence);

% store cats as strings for output.
predicted_categories = cats(indices);
end