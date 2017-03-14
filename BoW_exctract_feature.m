function [ features ] = BoW_exctract_feature( image, type )
if(strcmp(type,'keyPoint'))
elseif(strcmp(type,'dense'))
elseif(strcmp(type,'rgb'))
elseif(strcmp(type,'normRGB'))
elseif(strcmp(type,'opponent'))
end

end

