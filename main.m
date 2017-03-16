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
    disp('Build visual dictionnary');
    [train_descriptor_names, train_cls_names, test_cls_names] = preprocess_data(train_percentage, nb_train_each_class, nb_test_each_class);
    [visual_dic] = build_visual_vocab( train_descriptor_names, vocab_size, sift_type);
    
    %quantisize feature using a visual vocabulary.
    %randomly picked an image just to quantize and show its histogram
    disp('Quantisize one random image to build histogram');
    perm = randperm(size(train_descriptor_names,1));
    name = num2str(cell2mat(train_descriptor_names(perm(1))));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    [f,d]= BoW_exctract_feature(im, sift_type);
    visual_freq = quantize_feature(visual_dic, d, vocab_size);
    
    %representing image by frequencies of visual words
    % TODO : not sure whether this is the right way to plot histogram
    disp('Show histogram of random image');
    figure
    histogram(visual_freq)
    
    %classification
    for obj = 1:4
        disp(strcat('Create classifier of object ',num2str(obj),'/4'));
        % build X(histogram of visual word) and y(positive or negative) matrix to train an svm
        X = zeros ([length(train_cls_names) vocab_size]);
        y = zeros ([length(train_cls_names) 1]);
        for i = 1: length(train_cls_names)
            name = num2str(cell2mat(train_cls_names(i)));
            disp(strcat('Get features for_ ',name,'_step_',num2str(i),'/',num2str(length(train_cls_names)) ));
            im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
            [f,d]= BoW_exctract_feature( im, sift_type );
            X(i,:) = quantize_feature(visual_dic, d, vocab_size);
            if((obj-1) *nb_train_each_class < i && (i <= obj * nb_train_each_class))
                y(i) = 1;
            else
                y(i) = -1;
            end
        end
        
        %training the model
        if(obj == 1) 
            SVMModel_airp = fitcsvm(X,y);
        elseif(obj==2)
            SVMModel_cars = fitcsvm(X,y);
        elseif(obj==3)
            SVMModel_faces = fitcsvm(X,y);
        elseif(obj==4)
            SVMModel_motor = fitcsvm(X,y);
        end
    end
    
    %testing:
    %build matrix for testing
    disp(strcat('Test classifier ','1','/4'));
    X_test = zeros ([length(test_cls_names) vocab_size]);
    for i = 1: length(test_cls_names)
        name = num2str(cell2mat(test_cls_names(i)));
        im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
        [f,d]= BoW_exctract_feature( im, sift_type );
        X_test(i,:) = quantize_feature(visual_dic, d, vocab_size);
    end
    
    label = predict(SVMModel_airp,X_test);
    
    %compute Mean average precision
    disp('Compute MaP');
elseif(strcmp(test,'CNN'))
    %define network architecture
    %Data preprocessing
    %Data feeding
    
    %testing:
    %visualize feature space
    %evaluate accuracy
end