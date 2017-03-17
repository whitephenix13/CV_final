function [visual_freq] = quantize_feature(visual_dic, descriptor, vocab_size)
visual_freq = zeros([vocab_size 1]);
%for all the descriptor in the image 
min_dist = pdist2(double(visual_dic), double(descriptor'));
[value index] = min(min_dist);
for i=1:vocab_size
    visual_freq(i) = sum(index==i);
end
end