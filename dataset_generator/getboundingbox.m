function [BB] = getboundingbox(A)
%��ȡͼƬ�����ֵ�boundingboxλ��
%   �˴���ʾ��ϸ˵��
ymax=64-Ceil(rot90(A,2))+1;%should be 28
xmax=64-Ceil(rot90(A))+1;
ymin=Ceil(A);
xmin=Ceil(rot90(A,3));
BB=[xmin,ymin,xmax,ymax];
end

