function [ ap ] = averagePrecision( labels, desired_output )
    %ranked the labels 
    [ranked,r_index]= sort(labels,'descending');
    %rerank desired output: 
    rerank_desired = desired_output(r_index);
    
    num_pos = 0;
    sum = 0;
    for i=1:length(rerank_desired)
        if(rerank_desired(i)==1)
            num_pos= num_pos + 1;
            sum = sum + (num_pos / i);
        end
    end
    ap = sum/(num_pos);
end

