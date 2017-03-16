function [visual_freq] = quantize_feature(visual_dic, descriptor, vocab_size)
visual_freq = zeros([vocab_size 1]);
for i=1:size(descriptor,2)
    min_distance = 1000000000;
    descriptor_centroid = 0;
    for j=1:size(visual_dic,1)
        dist = pdist2(double(descriptor(:,i)'),double(visual_dic(j,:)));
        if(dist < min_distance)
            min_distance = dist;
            descriptor_centroid = j;
        end
    end
    visual_freq(descriptor_centroid) = visual_freq(descriptor_centroid) + 1;
end
end