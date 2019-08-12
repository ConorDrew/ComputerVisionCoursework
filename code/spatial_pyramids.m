function [image_feats] = spatial_pyramids(img, stepSize,...
    binSize, colorSpace, vocab)

vocab_size = size(vocab, 1); 
[rows, ~] = size(img); 
[columns] = size(img, 1);
col1 = 1;
col2 = floor(columns/2);
col3 = col2 + 1;
row1 = 1;
row2 = floor(rows/2);
row3 = row2 + 1;

upperLeft = imcrop(img, [row3 col1 rows col2]);
upperRight = imcrop(img, [row3 rows col3 columns]);
lowerLeft = imcrop(img, [row1 col1 row2 col2]);
lowerRight = imcrop(img, [row1 col3 row2 columns]);

image_feats = zeros(5, vocab_size);

[image_feats(1,:)] = spatial_pyramids_histcount(img, stepSize, binSize, colorSpace, vocab);
[image_feats(2,:)]  = spatial_pyramids_histcount(upperLeft, stepSize, binSize, colorSpace, vocab);
[image_feats(3,:)]  = spatial_pyramids_histcount(upperRight, stepSize, binSize, colorSpace, vocab);
[image_feats(4,:)]  = spatial_pyramids_histcount(lowerLeft, stepSize, binSize, colorSpace, vocab);
[image_feats(5,:)]  = spatial_pyramids_histcount(lowerRight, stepSize, binSize, colorSpace, vocab);

end

