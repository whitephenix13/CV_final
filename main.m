%run('vlfeat-0.9.20/toolbox/vl_setup')

load_vis = true;
load_cla = true;
load_lab = true;

save_visual_dict = false;
save_classifier=false;
save_labels= false;
%CNN variables
%...
tic;
%im = imread('Caltech4/ImageData/airplanes_test/img001.jpg');

vocab_size = 400;%400, 800, 1600
train_percentage = 0.5;%0.5
color_type = 'intensity'; %intensity rgb, normRGB , opponent
type = 'dense';%keyPoint dense
nb_train_each_class = 50;%50 200 400
kernel_type = 'linear';%linear rbf polynomial

%fill if kernel_type is polynomial
kernel_type_order =3;

sift_type=strcat(color_type,type);
load_visual_dict='';%Caltech4\FeatureData\visual_dict_400_0.5_keyPoint.mat
load_classifier='';%Caltech4\FeatureData\SVMModel_400_0.5_keyPoint.mat
load_labels='';%Caltech4\FeatureData\labels_400_0.5_keyPoint.mat
if(load_vis)
    load_visual_dict=strcat('Caltech4/FeatureData/visual_dict_',num2str(vocab_size),'_',num2str(train_percentage)...
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
end
if(load_cla)
    load_classifier=strcat('Caltech4/FeatureData/SVMModel_',num2str(vocab_size),'_',num2str(train_percentage)...
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
end
if(load_lab)
    load_labels=strcat('Caltech4/FeatureData/labels_',num2str(vocab_size),'_',num2str(train_percentage)...
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
end

sift_type= strcat(color_type,type);
nb_test_each_class = 50;

%build a visual vocabulary
if(strcmp(load_visual_dict,''))
    disp('Preprocess data');
    [train_descriptor_names, train_cls_names, test_cls_names] = preprocess_data(train_percentage, nb_train_each_class, nb_test_each_class);
    disp('Build visual dictionnary');
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
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
    save(fname,'train_descriptor_names','train_cls_names','test_cls_names','visual_dic');
end
toc
%quantisize feature using a visual vocabulary.
%randomly picked an image just to quantize and show its histogram
disp('Quantisize one random image to build histogram');
d=-1;
while(d==-1)
    perm = randperm(size(train_descriptor_names,1));
    name = num2str(cell2mat(train_descriptor_names(perm(1))));
    im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
    [ignore,f,d]= BoW_exctract_feature(im, sift_type);
end
if(~ignore)
    visual_freq = quantize_feature(visual_dic, d, vocab_size);
    size(d)
    size(visual_freq)
    
    %representing image by frequencies of visual words
    %normalise hitstogram:
    visual_freq = visual_freq ./ (sum(visual_freq));
    disp('Show histogram of random image');
    figure
    bar(visual_freq)
end
toc
%classification
if(strcmp(load_classifier,''))
    for obj = 1:4
        disp(strcat('Create classifier of object ',num2str(obj),'/4'));
        % build X(histogram of visual word) and y(positive or negative) matrix to train an svm
        X = zeros ([length(train_cls_names) vocab_size]);
        y = zeros ([length(train_cls_names) 1]);
        ind = 1;
        for i = 1: length(train_cls_names)
            name = num2str(cell2mat(train_cls_names(i)));
            disp(strcat('Get features for_ ',name,'_step_',num2str(i),'/',num2str(length(train_cls_names)) ));
            im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
            [ignore,f,d]= BoW_exctract_feature( im, sift_type );
            if(~ignore)
                X(ind,:) = quantize_feature(visual_dic, d, vocab_size);
                if((obj-1) *nb_train_each_class < i && (i <= obj * nb_train_each_class))
                    y(ind) = 1;
                else
                    y(ind) = -1;
                end
                ind = ind +1;
            end
        end
        X(ind:size(X,1),:)=[];
        y(ind:size(y,1),:)=[];
        %training the model
        if(strcmp(kernel_type, 'polynomial'))
            svm_res = fitcsvm(X,y,'KernelFunction',kernel_type,'PolynomialOrder',kernel_type_order);
        else
            svm_res = fitcsvm(X,y,'KernelFunction',kernel_type);
        end
        if(obj == 1)
            SVMModel_airp = svm_res;
        elseif(obj==2)
            SVMModel_cars = svm_res;
        elseif(obj==3)
            SVMModel_faces = svm_res;
        elseif(obj==4)
            SVMModel_motor = svm_res;
        end
    end
else
    vars = load(load_classifier);
    SVMModel_airp=vars.SVMModel_airp;
    SVMModel_cars=vars.SVMModel_cars;
    SVMModel_faces=vars.SVMModel_faces;
    SVMModel_motor=vars.SVMModel_motor;
end
if(save_classifier)
    fname = strcat('Caltech4\FeatureData\SVMModel_',num2str(vocab_size),'_',num2str(train_percentage)...
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
    save(fname,'SVMModel_airp','SVMModel_cars','SVMModel_faces','SVMModel_motor');
end
toc
%testing:
%build matrix for testing
labels = zeros(length(test_cls_names),4);
Y_test = zeros(length(test_cls_names),4);

if(strcmp(load_labels,''))
    for obj=1:4
        disp(strcat('Test classifier ',num2str(obj),'/4'));
        X_test = zeros ([length(test_cls_names) vocab_size]);
        ind=1;
        for i = 1: length(test_cls_names)
            name = num2str(cell2mat(test_cls_names(i)));
            im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
            [ignore,f,d]= BoW_exctract_feature( im, sift_type );
            if(~ignore)
                X_test(ind,:) = quantize_feature(visual_dic, d, vocab_size);
                if((obj-1) *nb_test_each_class < i && (i <= obj * nb_test_each_class))
                    Y_test(ind,obj) = 1;
                else
                    Y_test(ind,obj) = 0;
                end
                ind = ind +1;
            end
        end
        %resizing X and Y and labels
        X_test(ind:size(X_test,1),:)=[];
        Y_test(ind:size(Y_test,1),:)=[];
        labels(ind:size(labels),:)=[];
        %if lab = 1 ; -1  then score might be  -1.2 1.2 ; 0.9 -0.9
        if(obj == 1)
            [lab,score] = predict(SVMModel_airp,X_test);
            labels(:,obj) = score(:,2);
        elseif(obj==2)
            [lab,score] = predict(SVMModel_cars,X_test);
            labels(:,obj) = score(:,2);
        elseif(obj==3)
            [lab,score] = predict(SVMModel_faces,X_test);
            labels(:,obj) = score(:,2);
        elseif(obj==4)
            [lab,score] = predict(SVMModel_motor,X_test);
            labels(:,obj) = score(:,2);
        end
    end
    
else
    vars = load(load_labels);
    labels=vars.labels;
    Y_test=vars.Y_test;
end
if(save_labels)
    fname = strcat('Caltech4\FeatureData\labels_',num2str(vocab_size),'_',num2str(train_percentage)...
        ,'_',num2str(sift_type),'_',num2str(nb_train_each_class),'_',kernel_type,'.mat');
    save(fname,'labels','Y_test');
end
%loop over all the labels to compute the average precision
ap_airp=averagePrecision(labels(:,1),Y_test(:,1));
disp(strcat('AP for airplane: ',num2str(ap_airp)));

ap_cars=averagePrecision(labels(:,2),Y_test(:,2));
disp(strcat('AP for cars: ',num2str(ap_cars)));

ap_face=averagePrecision(labels(:,3),Y_test(:,3));
disp(strcat('AP for faces: ',num2str(ap_face)));

ap_moto=averagePrecision(labels(:,4),Y_test(:,4));
disp(strcat('AP for motorbikes: ',num2str(ap_moto)));
%compute Mean average precision
toc
disp(strcat('MaP =_',num2str((ap_airp+ap_cars+ap_face+ap_moto)/4)));

%Retrieve the test image ranked
retrieve_ranking = true;
if(retrieve_ranking)
    %Do we consider Gray images in the testing?
    testGray = true;
    test__names=test_cls_names;
    
    if(~testGray)
        test__names={};
        for i = 1: length(test_cls_names)
            name = num2str(cell2mat(test_cls_names(i)));
            im = imread(strcat('Caltech4/ImageData/',name,'.jpg'));
            if(size(im,3)==3)
                test__names=[test__names;name];
            end
        end
    end
    
    [ranked,r1_index]= sort(labels(:,1),'descend');
    [ranked,r2_index]= sort(labels(:,2),'descend');
    [ranked,r3_index]= sort(labels(:,3),'descend');
    [ranked,r4_index]= sort(labels(:,4),'descend');
    for j=1:length(test__names)
        airp_name=test__names{r1_index(j)};
        car_name=test__names{r2_index(j)};
        face_name=test__names{r3_index(j)};
        motor_name=test__names{r4_index(j)};
        
        html_line =strcat('<tr><td><img src="Caltech4/ImageData/', ...
            airp_name,'.jpg" /></td><td><img src="Caltech4/ImageData/', ...
            car_name, '.jpg" /></td><td><img src="Caltech4/ImageData/',...
            face_name,'.jpg" /></td><td><img src="Caltech4/ImageData/',...
            motor_name,'.jpg" /></td></tr>');
        disp(html_line);
    end
end


