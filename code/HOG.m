train_image_feats = train_image_feats;
train_labels = train_labels;
test_labels = test_labels;
categories = categories;
test_image_feats = test_image_feats;
image_paths = train_image_paths;

clearvars -except train_image_feats train_labels test_image_feats test_labels...
    categories image_paths vocab_size train_image_paths

%%

img1 = imread(image_paths{1});
img2 = imread(image_paths{101});
img3 = imread(image_paths{201});

imshow(img2);

redChannel = img1(:, :, 1);
greenChannel = img1(:, :, 2);
blueChannel = img1(:, :, 3);

meanR = mean(redChannel);
meanG = mean(greenChannel);
meanB = mean(blueChannel);

[binR,~] = imhist(redChannel, 16);
