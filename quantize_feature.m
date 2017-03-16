function [visual_freq] = quantize_feature(visual_dic, descriptor, vocab_size)
visual_freq = zeros([vocab_size 1]);
%for all the descriptor in the image 
total_num = size(descriptor,2) * size(visual_dic,1);
for i=1:size(descriptor,2)
    min_distance = 1000000000;
    descriptor_centroid = 0;
    %look for the the closest visual word for this descriptor 
    for j=1:size(visual_dic,1)
        current_ind = i + (j-1) * size(descriptor,2);
        if(mod(current_ind,10000)==0)
            %disp(strcat('step_ ', num2str(current_ind) , '/', num2str(total_num)));
        end
        dist = pdist2(double(descriptor(:,i)'),double(visual_dic(j,:)));
        if(dist < min_distance)
            min_distance = dist;
            descriptor_centroid = j;
        end
    end
    visual_freq(descriptor_centroid) = visual_freq(descriptor_centroid) + 1;
end
end