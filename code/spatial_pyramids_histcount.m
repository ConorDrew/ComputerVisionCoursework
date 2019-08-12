function [image_feat] = spatial_pyramids_histcount(img, stepSize,...
    binSize, colorSpace, vocab)
vocab_size = size(vocab, 1); % Gets dimension of vocab.mat

[frames, features] = vl_phow(img, 'Sizes', binSize, 'Fast', 'True', 'Step', stepSize, 'Color', colorSpace);
features = single(features);

[indices, ~] = knnsearch(vocab, features', 'K', 3);
    
[image_feat, ~] = histcounts(indices, vocab_size);


end

