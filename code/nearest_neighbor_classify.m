function [output] = nearest_neighbor_classify(k,train_data, train_labels, test_data)
% nearest_neighbor_classify: Classifying using k-nearest neighbors algorithm
% Knn uses Euclidean Distance
% INPUT:
%   K: amount of nearest neighbors
%   TRAIN_DATA: (NxD) Matrix, N = number of entries, D = Dimensions of each
%   data point,
%   TRAIN_LABELS: (Nx1) Matrix, N = number of entries, 1 = label catagorie
%   TEST_DATA: (NxD) Matrix, N = number of entries, D = Dimensions of each
%   data point
%
% OUTPUT:
%   OUTPUT: Predicited labels based on the Knn algorithm
%

%% Faster eculdian dist.

% New ecudlead dist using VL_Feat toolbox 
distances = vl_alldist2(train_data', test_data');
distances = distances';
% Store sorted dist in dSorted and index in smallestIndex
[dSorted,smallestIndex] = sort(distances, 2);
train_labels = string(train_labels); % convert cell to string
testDataCol = length(test_data(:,1));
prediction = strings(testDataCol, 1);
kIndex = smallestIndex(:,1:k);

% psudeo
% get the label from train_Labels(KIndex) from all KIndex
% find majority vote of labels
% store predicition.

for i = 1:length(kIndex(:,1))
    labels = unique(train_labels(kIndex(i,:)'));
    
    largestValue = 0;
    %largestLabel = 0;
    for j = 1:length(labels)
        amount = length(find(train_labels(kIndex(i,:)')==labels(j)));
        if amount > largestValue
            largestValue = amount;
            largestLabel = labels(j);
        end
    end
    prediction(i) = largestLabel;
    
end

output = cellstr(prediction);


end
