function [union] = splice(Size,pic,BB,scale,presplice,anchor)
%��ԭͼճ����ȫ�㱳����
%   �˴���ʾ��ϸ˵��
dim=size(pic);
buffer=zeros(2,Size,Size);
buffer(1,:,:)=presplice;
buffer(2,:,:)=double([zeros(anchor(2)-1,Size);zeros(dim(1),anchor(1)-1),pic,zeros(dim(1),Size-anchor(1)-dim(2)+1);zeros(Size-anchor(2)-dim(1)+1,Size)]);
buffer=max(buffer);
%aftersplice=presplice+double([zeros(anchor(2)-1,Size);zeros(dim(1),anchor(1)-1),pic,zeros(dim(1),Size-anchor(1)-dim(2)+1);zeros(Size-anchor(2)-dim(1)+1,Size)]);
aftersplice=zeros(Size,Size);
aftersplice(:,:)=buffer(1,:,:);

newBB=int8([anchor(1),anchor(2),anchor(1),anchor(2)]+scale.*(BB-[1,1,0,0]));
union=cell(1,2);
union{1}=aftersplice;
union{2}=newBB;
end

