function [BB] = getboundingbox(A)
%获取图片中数字的boundingbox位置
%   此处显示详细说明
ymax=64-Ceil(rot90(A,2))+1;%should be 28
xmax=64-Ceil(rot90(A))+1;
ymin=Ceil(A);
xmin=Ceil(rot90(A,3));
BB=[xmin,ymin,xmax,ymax];
end

