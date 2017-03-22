function [train_descriptor_names, train_cls_names, test_cls_names] = preprocess_data(train_percent_size, nb_train_each_class, nb_test_each_class)

%get the names of the file we want to train the model on
folder = 'Caltech4/ImageSets/';
filenames = dir(strcat(folder,'*_train.txt'));
train_descriptor_names = {}; 
train_cls_names={};
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
    end
    t_names= t_names(randperm(linenum));
    % retrieve percentage images to cluster descriptors
    t_size= int64(train_percent_size*linenum);
    train_descriptor_names=[train_descriptor_names;t_names(1:t_size)];

    % retrieve n images from each that is not used to cluster
    % descriptor
    random_im = randsample(t_names(t_size+1:length(t_names)), nb_train_each_class);
    train_cls_names=[train_cls_names;random_im];
end

% get test filenames
filenames = dir(strcat(folder,'*_test.txt'));
test_cls_names={};
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
    end
    % retrieve n test images
    test_im = randsample(t_names, nb_test_each_class);
    test_cls_names=[test_cls_names;test_im];
end

end