function [B] = randdel(A,mode,num_in_one,flag)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
if (mode==0)
    B=A;
else
if (num_in_one==(2*mode+1))
    B=[A(1:flag-1,:);A(flag+1:end,:)];
else
    B=A;
end
end
end

