%run('vlfeat-0.9.20/toolbox/vl_setup')
test='BoW'; % BoW CNN
%BOW variables
%if a name is specified, load the corresponding dictionnary
%load_visual_dict='Caltech4/FeatureData/visual_dict_50_0.02_keyPoint.mat';
load_visual_dict='';
load_classifier='';
load_labels='';

save_visual_dict = false;
save_classifier=false;
save_labels= false;
%CNN variables
%...
if(strcmp(test,'BoW'))
    im = imread('Caltech4/ImageData/airplanes_test/img001.jpg');
    vocab_size = 200;
    train_percentage = 0.02;
    sift_type = 'dense';
    %
    nb_train_each_class = 3;
    nb_test_each_class = 2;
    
    %build a visual vocabulary
    disp('Build visual dictionnary');
    if(strcmp(load_visual_dict,''))
        [train_descriptor_names, train_cls_names, test_cls_names] = preprocess_data(train_percentage, nb_train_each_class, nb_test_each_class);
        [visual_dic] = build_visual_vocab( train_descriptor_names, vocab_size, sift_type);
    else
        vars = load(load_visual_dict);
        visual_dic=vars.visual_dic;
        train_descriptor_names=vars.train_descriptor_names;
        train_cls_names=vars.train_cls_names;
        test_cls_names=vars.test_cls_names;
    end
    if(save_visual_dict)
        fname = strcat('Caltech4\FeatureData\visual_dict_',num2str(vocab_size),'_',num2str(train_percentage)...
            ,'_',num2str(sift_type),'.mat');
        save(fname,'train_descriptor_names','train_cls_names','test_cls_names','visual_dic');
    end
    %quantisize feature using a visual vocabulary.
    %randomly picked an image just to quantize and show its histogram
    disp('Quantisize one random image to build histogram');
    perm = randperm(size(train_descriptor_names,1));
    name = num2str(cell2mat(train_descriptor_names(perm(1))));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    [f,d]= BoW_exctract_feature(im, sift_type);
    visual_freq = quantize_feature(visual_dic, d, vocab_size);
    size(d)
    size(visual_freq)
    
    %representing image by frequencies of visual words
    %normalise hitstogram:
    visual_freq = visual_freq ./ (sum(visual_freq));
    disp('Show histogram of random image');
    figure
    bar(visual_freq)
    
    %classification
    for obj = 1:1 %TODO:4
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
    for obj=1:1%TODO: 4
        disp(strcat('Test classifier ',num2str(obj),'/4'));
        %TODO: shuffle X_test before (to really test performance)
        X_test = zeros ([length(test_cls_names) vocab_size]);
        Y_test = zeros(length(test_cls_names),1);
        for i = 1: length(test_cls_names)
            name = num2str(cell2mat(test_cls_names(i)));
            im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
            [f,d]= BoW_exctract_feature( im, sift_type );
            X_test(i,:) = quantize_feature(visual_dic, d, vocab_size);
            if((obj-1) *nb_test_each_class < i && (i <= obj * nb_test_each_class))
                Y_test(i) = 1;
            else
                Y_test(i) = -1;
            end
        end
        if(obj == 1)
            label = predict(SVMModel_airp,X_test);
        elseif(obj==2)
            label = predict(SVMModel_cars,X_test);
        elseif(obj==3)
            label = predict(SVMModel_faces,X_test);
        elseif(obj==4)
            label = predict(SVMModel_motor,X_test);
        end
        %compute Mean average precision
        disp(averagePrecision(label,Y_test))
    end
    %TODO compute mean average precision
    disp('Compute MaP');
elseif(strcmp(test,'CNN'))
    %define network architecture
    %Data preprocessing
    %Data feeding
    
    %testing:
    %visualize feature space
    %evaluate accuracy
end