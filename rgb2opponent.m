function [output_image] = rgb2opponent(input_image)
% converts an RGB image into opponent color space
R = input_image(:,:,1);
G = input_image(:,:,2);
B = input_image(:,:,3);
O1 = (R -G) / sqrt(2);
O2 = (R + G -2 * B) / sqrt(6);
O3 = (R + G + B) / sqrt(3);
output_image = O1;
output_image(:,:,2) = O2;
output_image(:,:,3) = O3;
end

