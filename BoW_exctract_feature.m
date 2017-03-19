function [ frames, descrip ] = BoW_exctract_feature( image, type )
im = im2single(image);
if(size(im,3)>1)
    im = rgb2gray(im);
end
if(strcmp(type,'keyPoint'))
    %http://www.vlfeat.org/matlab/vl_sift.html
    %descriptor meaning: https://www.inf.fu-berlin.de/lehre/SS09/CV/uebungen/uebung09/SIFT.pdf
    
    [frames,descrip] = vl_sift(im);
    %each column of frame is a feature frame of format [X;Y;S;TH], (x,y)
    %center of frame, S scale, TH orientation in radian
    %descrip= descriptor: 4 * 4 * 8 = 128, histogram of direction 
elseif(strcmp(type,'dense'))
    %http://www.vlfeat.org/matlab/vl_dsift.html
    
    %should we smooth the image ? it seems only usefull if we want to
    %archieve the same result as vl_shift
    
    %calculate descriptors
    bin_size =98;%12
    [frames,descrip] = vl_dsift(im,'size',bin_size);
    %frames is a 2 x num_keypoints matrix which reprensents the center(x,y) of the keypoint frame. 
    %descrip is a 128 x num_keypoints matrix with 1 descriptor per
    %column, same format as vl_sift
elseif(strcmp(type,'rgb'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
    else
       [frames,descrip]= vl_phow(single(image), 'Color', 'rgb','Sizes',bin_size);
    end
elseif(strcmp(type,'normRGB'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
    else
        image_norm_rgb = im2single(rgb2normedrgb(image));
        [frames,descrip]= vl_phow(image_norm_rgb, 'Color', 'rgb','Sizes',bin_size);
    end

elseif(strcmp(type,'opponent'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
    else
       [frames,descrip]= vl_phow(im2single(image), 'Color', 'opponent','Sizes',bin_size);
    end
end

end

