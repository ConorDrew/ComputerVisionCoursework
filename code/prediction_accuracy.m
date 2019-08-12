function accuracy = prediction_accuracy(prediction_categories, categories, test_labels)
%PREDICTION_ACCURACY Summary of this function goes here
%   Detailed explanation goes here

predicted_categories = prediction_categories;
num_categories = length(categories);
confusion_matrix = zeros(num_categories, num_categories);
for i=1:length(predicted_categories)
    row = find(strcmp(test_labels{i}, categories));
    column = find(strcmp(predicted_categories{i}, categories));
    confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
end
%if the number of training examples and test casees are not equal, this
%statement will be invalid.
num_test_per_cat = length(test_labels) / num_categories;
confusion_matrix = confusion_matrix ./ num_test_per_cat;   
accuracy = mean(diag(confusion_matrix));
% Print for debug
%fprintf(     'Accuracy (mean of diagonal of confusion matrix) is %.3f\n', accuracy)

end

