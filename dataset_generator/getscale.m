function [scale] = getscale(mode,num_in_one)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
if (mode==0)
    scale=14/28+(46/28)*rand;
else
if (mode==1)
    scale=14/28+(30/28)*rand(1,num_in_one);
else
if (mode==2)
    scale=14/28+(13/28)*rand(1,num_in_one);
end
end
end
end

