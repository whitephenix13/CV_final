function [ ignore,frames, descrip ] = BoW_exctract_feature( image, type )
ignore = false;
im = im2single(image);
if(size(im,3)>1)
    im = rgb2gray(im);
end
if(strcmp(type,'intensitykeyPoint'))
    %http://www.vlfeat.org/matlab/vl_sift.html
    %descriptor meaning: https://www.inf.fu-berlin.de/lehre/SS09/CV/uebungen/uebung09/SIFT.pdf
    
    [frames,descrip] = vl_sift(im);
    %each column of frame is a feature frame of format [X;Y;S;TH], (x,y)
    %center of frame, S scale, TH orientation in radian
    %descrip= descriptor: 4 * 4 * 8 = 128, histogram of direction
elseif(strcmp(type,'intensitydense'))
    %http://www.vlfeat.org/matlab/vl_dsift.html
    
    %should we smooth the image ? it seems only usefull if we want to
    %archieve the same result as vl_shift
    
    %calculate descriptors
    bin_size =98;%12
    [frames,descrip] = vl_dsift(im,'size',bin_size);
    %frames is a 2 x num_keypoints matrix which reprensents the center(x,y) of the keypoint frame.
    %descrip is a 128 x num_keypoints matrix with 1 descriptor per
    %column, same format as vl_sift
elseif(strcmp(type,'rgbdense'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        [frames,descrip]= vl_phow(single(image), 'Color', 'rgb','Sizes',bin_size);
    end
elseif(strcmp(type,'rgbkeyPoint'))
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        %find keypoints
        [frames,descrip] = vl_sift(im);
        %for each color channel, by using the same frames, compute
        %descriptors separately
        s_image = single(image);
        descrip1 = vl_sift(s_image(:,:,1),'frames',frames);
        descrip2 = vl_sift(s_image(:,:,2),'frames',frames);
        descrip3 = vl_sift(s_image(:,:,3),'frames',frames);
        %concatenate descriptors
        descrip=[descrip1;descrip2;descrip3];
    end
elseif(strcmp(type,'normRGBdense'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        image_norm_rgb = single(rgb2normedrgb(image));
        [frames,descrip]= vl_phow(image_norm_rgb, 'Color', 'rgb','Sizes',bin_size);
    end
elseif(strcmp(type,'normRGBkeyPoint'))
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        %find keypoints
        [frames,descrip] = vl_sift(im);
        %convert image to new colorspace
        image_norm_rgb = single(rgb2normedrgb(image));
        %for each color channel, by using the same frames, compute
        %descriptors separately
        descrip1 = vl_sift(image_norm_rgb(:,:,1),'frames',frames);
        descrip2 = vl_sift(image_norm_rgb(:,:,2),'frames',frames);
        descrip3 = vl_sift(image_norm_rgb(:,:,3),'frames',frames);
        %concatenate descriptors
        descrip=[descrip1;descrip2;descrip3];
    end
    
elseif(strcmp(type,'opponentdense'))
    bin_size =94;
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        [frames,descrip]= vl_phow(single(image), 'Color', 'opponent','Sizes',bin_size);
    end
elseif(strcmp(type,'opponentkeyPoint'))
    if(size(image,3)<3)
        frames=-1;
        descrip = -1;
        ignore=true;
    else
        %find keypoints
        [frames,descrip] = vl_sift(im);
        %convert image to new colorspace
        image_opponent = single(rgb2opponent(image));
        %for each color channel, by using the same frames, compute
        %descriptors separately
        descrip1 = vl_sift(image_opponent(:,:,1),'frames',frames);
        descrip2 = vl_sift(image_opponent(:,:,2),'frames',frames);
        descrip3 = vl_sift(image_opponent(:,:,3),'frames',frames);
        %concatenate descriptors
        descrip=[descrip1;descrip2;descrip3];
    end
else
    disp(strcat('case not handled: ',type));
end

end

