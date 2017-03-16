%run('vlfeat-0.9.20/toolbox/vl_setup')
test='BoW'; % BoW CNN
if(strcmp(test,'BoW'))
    im = imread('Caltech4/ImageData/airplanes_test/img001.jpg');
    vocab_size = 400;
    train_percentage = 0.02;
    sift_type = 'keyPoint';
    % 
    nb_train_each_class = 3;
    nb_test_each_class = 2;
    
    %build a visual vocabulary
    [train_descriptor_names, train_cls_names, test_cls_names] = preprocess_data(train_percentage, nb_train_each_class, nb_test_each_class);
    [visual_dic] = build_visual_vocab( train_descriptor_names, vocab_size, sift_type);
    
    %quantisize feature using a visual vocabulary.
    %randomly picked an image just to quantize and show its histogram
    perm = randperm(size(train_descriptor_names,1));
    name = num2str(cell2mat(train_descriptor_names(perm(1))));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    [f,d]= BoW_exctract_feature(im, sift_type);
    visual_freq = quantize_feature(visual_dic, d, vocab_size);
    
    %representing image by frequencies of visual words
    % TODO : not sure whether this is the right way to plot histogram
    figure
    histogram(visual_freq)
    
    %classification
    % build X and y matrix to train an svm
    X = zeros ([length(train_cls_names) vocab_size]);
    y = zeros ([length(train_cls_names) 1]);
    for i = 1: length(train_cls_names)
        name = num2str(cell2mat(train_cls_names(i)));
        im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
        [f,d]= BoW_exctract_feature( im, sift_type );
        X(i,:) = quantize_feature(visual_dic, d, vocab_size);
        % TODO : still dirty, use aeroplane as positive label and the rest are
        % negative label
        if(i <= nb_train_each_class)
            y(i) = 1;
        else
            y(i) = -1;
        end
    end
    
    %training the model
    SVMModel = fitcsvm(X,y);
    
    %testing: 
    %build matrix for testing
    X_test = zeros ([length(test_cls_names) vocab_size]);
    for i = 1: length(test_cls_names)
        name = num2str(cell2mat(test_cls_names(i)));
        im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
        [f,d]= BoW_exctract_feature( im, sift_type );
        X_test(i,:) = quantize_feature(visual_dic, d, vocab_size);
    end
    
    label = predict(SVMModel,X_test);
    
    %compute Mean average precision
    
elseif(strcmp(test,'CNN'))
    %define network architecture
    %Data preprocessing 
    %Data feeding 
    
    %testing: 
    %visualize feature space
    %evaluate accuracy
end