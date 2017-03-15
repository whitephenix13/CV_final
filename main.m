run('vlfeat-0.9.20/toolbox/vl_setup')
test='BoW'; % BoW CNN
if(strcmp(test,'BoW'))
    im = imread('Caltech4\ImageData\airplanes_test\img001.jpg');
    %extract features 
    [ frames, descrip ] = BoW_exctract_feature( im, 'normRGB' );
    size(descrip)
    %build a visual vocabulary
    %quantisize feature using a visual vocabulary
    %representing image by frequencies of visual words
    %classification
    
    %testing: 
    %compute Mean average precision
elseif(strcmp(test,'CNN'))
    %define network architecture
    %Data preprocessing 
    %Data feeding 
    
    %testing: 
    %visualize feature space
    %evaluate accuracy
end