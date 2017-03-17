function [ visual_dic ] = build_visual_vocab( train_names, vocab_size, sift_type )
descriptors_concat = [];
for  i= 1: length(train_names)
    if(mod(i,100) == 0)
        disp(strcat(num2str(i),'/',num2str(length(train_names))));
    end
    name = num2str(cell2mat(train_names(i)));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    [frames,descript]= BoW_exctract_feature( im, sift_type );
    descriptors_concat=[descriptors_concat,descript];
end
disp('Ended feature extraction for visual vocab')
[idx,C]=kmeans(double(descriptors_concat'),double(vocab_size));
visual_dic=C;
end

