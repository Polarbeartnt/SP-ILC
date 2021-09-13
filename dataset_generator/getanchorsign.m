function [anchorsign] = getanchorsign(mode)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
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

