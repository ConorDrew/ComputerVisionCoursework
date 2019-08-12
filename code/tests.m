train_image_feats = train_image_feats;
train_labels = train_labels;
test_labels = test_labels;
categories = categories;
test_image_feats = test_image_feats;

clearvars -except train_image_feats train_labels test_image_feats test_labels...
    categories
%%


cats = unique(train_labels, 'stable'); % gets labels for each categorie.
num_cats = length(cats); % Gets number of categories

num_train = size(train_image_feats, 1); % get number of training data
num_test = size(test_image_feats, 1); % gets number of test data

featSize = size(test_image_feats, 2); % gets size of feature to store
wStored = zeros(num_cats, featSize); % Stores the values of W 
bStored = zeros(num_cats, 1); % Store the values of B
labels = zeros(num_train,1); % Makes a list to store labels

%% Accuracy Testing main

LAM_STEP = 10;
%LAMBDA = 0.000001;
LAMBDA = 10;
numIterations = 1000000;

lam_its = 1; % how many times lambda will increase
num_runs = 50; % how many times each lambda will run.
accuracyResult = zeros(lam_its, num_runs + 4); % adding 4 for lambda, avg, min and max


for lam_loop=1:lam_its % This loop will go though the step size
    for lam_avg=1:num_runs % This loop will run 10 times to get an avg
        % SVM Loop %This loop is for the real SVM 
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
        accuracy = prediction_accuracy(predicted_categories, categories, test_labels);
        accuracyResult(lam_loop, lam_avg+1) = accuracy;
    end
    accuracyResult(lam_loop, 1) = LAMBDA;
    accuracyResult(lam_loop, lam_avg+2) = mean(accuracyResult(lam_loop, 2:lam_avg+1), 2);
    accuracyResult(lam_loop, lam_avg+3) = min((accuracyResult(lam_loop, 2:lam_avg+1)));
    accuracyResult(lam_loop, lam_avg+4) = max((accuracyResult(lam_loop, 2:lam_avg+1)));
    LAMBDA = LAMBDA * LAM_STEP;
end

clearvars -except train_image_feats train_labels test_image_feats test_labels...
    categories accuracy accuracyResult

%output = [header; accuracyResult];
% output is accuracy results
% 1st col is the lambda or itterations value, 
% the last 3 are Avg, Min, Max 
% everything in the middle are diffrent test runs.
beep
pause(0.2)
beep
pause(0.2)
beep
pause(0.2)
beep
pause(0.2)
