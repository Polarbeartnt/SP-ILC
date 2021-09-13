function [anchor] = getrandanchor(Size,pic,mode)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
dim=size(pic);
if(mode==0)
    anchor=[randi(Size-dim(2)+1),randi(Size-dim(1)+1)];
else
if(mode==1)
    anchor=[randi(min(16,Size-dim(2)+1)),randi(min(16,Size-dim(1)+1))];
else
if(mode==2)
    anchor=[randi(min(10,Size-dim(2)+1)),randi(min(16,Size-dim(1)+1))];
end
end
end
end

