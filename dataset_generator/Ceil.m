function [ymax] = Ceil(pic)
%��ȡ�������϶˵�������
%   �˴���ʾ��ϸ˵��
ymax=0;
dim1=size(pic,1);
dim2=size(pic,2);
for i=1:dim2
    if(ymax>0)
        break;
    end
    for j=1:dim1
        if(pic(i,j)>=10)
            ymax=i;
            break;
        end
    end
end    
end

