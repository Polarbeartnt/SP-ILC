function [anchorsign] = getanchorsign(mode)
%UNTITLED6 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if (mode==0)
    anchorsign=[1,1];
else
if (mode==1)
    anchorsign=[1,1;1,-1;-1,1;-1,-1];
else
if (mode==2)
    anchorsign=[1,1;1,-1;-1,1;-1,-1;1,1;1,-1];
end
end
end
end

