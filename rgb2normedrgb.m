function [output_image] = rgb2normedrgb(input_image)
% converts an RGB image into normalized rgb
R = input_image(:,:,1);
G = input_image(:,:,2);
B = input_image(:,:,3);
sum = (R + G + B);
O1 = R ./sum  ;
O2 = G ./sum  ;
O3 = B ./sum  ;
output_image = O1;
output_image(:,:,2) = O2;
output_image(:,:,3) = O3;
end

