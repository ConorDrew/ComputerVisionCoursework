function [colorHist] = get_color_histogram(input_images, binSize, colorSpace)
% get_color_histogram: Create output based on color histogram
% Colour histograms creates a smaller dataset.
% 
% INPUT:
%   INPUT_IMAGES: (Nx1) Cell, N = Image paths.
%   QUANT_NUM: changes range of colour numbers, from deafult 8 bit image to
%   to the input.
% OUTPUT:
%   OUTPUT: colour histogram dataset.
%

    %% Var set
    validateattributes(input_images,{'cell'},{'vector'});

    colorHist = zeros(length(input_images),binSize*3);
    %% Main Conversion

    % Loops thought every image input.
    for i = 1:length(input_images)

        imgTemp = imread(char(input_images(i)));
        switch lower(colorSpace)    
          
            case 'lab'
                imgTemp = rgb2lab(imgTemp);
            case 'hsv'
                imgTemp = rgb2hsv(imgTemp);
            case 'ntsc'
                imgTemp = rgb2ntsc(imgTemp);
            case 'ycbcr'
                imgTemp = rgb2ycbcr(imgTemp);
            % Else it stays as RGB
        end
                

        Red = imgTemp(:,:,1);
        Green = imgTemp(:,:,2);
        Blue = imgTemp(:,:,3);

        [binR,~] = imhist(Red, binSize);
        [binG,~] = imhist(Green, binSize);
        [binB,~] = imhist(Blue, binSize);


        colorHist(i,:) = cat(2, binR', binG', binB');

    end
end

