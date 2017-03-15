function [ visual_dic ] = build_visual_vocab( train_percent_size, vocab_size, sift_type )
    %get the names of the file we want to train the model on
    folder = 'Caltech4\ImageSets\';
    filenames = dir(strcat(folder,'*_train.txt'));
    train_names = {};
    for i=1:length(filenames)
        file=fopen(strcat(folder,filenames(i).name),'r');
        t_names = {};
        linenum=0;
        EoF = false; 
        while(~EoF)
            line= fgets(file);
            if(line== -1)
                EoF = true;
            else
                t_names = [t_names;line];
                linenum=linenum + 1;
            end
            
            %if(linenum>5) 
               % EoF = true;
            %end
        end
        t_names= t_names(randperm(linenum));
        t_size= int64(train_percent_size*linenum);
        train_names=[train_names;t_names(1:t_size)];
    end
    descriptors_concat = [];
    for  i= 1: length(train_names)
        name = num2str(cell2mat(train_names(i)));
        im = imread(strcat('Caltech4\ImageData\',name,'.jpg'));
        [frames,descript]= BoW_exctract_feature( im, sift_type );
        descriptors_concat=[descriptors_concat,descript];
    end
    [idx,C]=kmeans(double(descriptors_concat'),double(vocab_size));
    visual_dic=C;
end

