function [ visual_dic ] = build_visual_vocab( train_names, vocab_size, sift_type )

ind = 1;
name = num2str(cell2mat(train_names(1)));
im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
clear name
[frames,descript]= BoW_exctract_feature( im, sift_type );
clear frames
descriptors_concat = zeros(size(descript,1),2500000);

descriptors_concat(:,ind:(ind+size(descript,2)-1))=descript;
ind= ind + size(size(descript,2));
clear descript;
for  i= 2: length(train_names)
    if(mod(i,100) == 0)
        disp(strcat(num2str(i),'/',num2str(length(train_names))));
    end
    name = num2str(cell2mat(train_names(i)));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    clear name
    [frames,descript]= BoW_exctract_feature( im, sift_type );
    clear frames
    descriptors_concat(:,ind:(ind+size(descript,2)-1))=descript;
    ind= ind + size(descript,2);
    clear descript;
end
disp('Ended feature extraction for visual vocab')
descriptors_concat(:,ind:size(descriptors_concat,2))=[];
size(descriptors_concat)
[idx,C]=kmeans(double(descriptors_concat'),double(vocab_size));
clear idx
visual_dic=C;
end

