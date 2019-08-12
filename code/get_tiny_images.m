function [output] = get_tiny_images(input, size)
% get_tiny_images: scale down image to a set size
% tiny images uses Nearest-neighbor interpolation (imresize)
% INPUT:
%   INPUT: (Nx1) Cell, N = Image paths.
%   SIZE: resize value will be (size x size)
% OUTPUT:
%   OUTPUT: (NxD) Matrix, N = Amount of entries, D = size^2

%% Var Setup

% check if input is cell datatype
validateattributes(input,{'cell'},{'vector'});

% Work out feature size from input
featSize = size*size;

% Create space for array based on feature size
outputTemp = zeros(length(input),featSize);

%%  Loop though every image path.
for i = 1:length(input)
    
    % Read char from cell and sdoc tore as image
    imgTemp = imread(char(input(i)));
    
    
    %% RGB
    imgTemp = imresize(imgTemp, [size size]);
    
    imgTemp = mean(imgTemp,3);
    
    %% Normalization
    
    %zscore it
    Z = zscore(imgTemp(:));
    
    % transpose it
    Z = Z';
    
    outputTemp(i,:) = Z;
    
end

output = outputTemp;
end

