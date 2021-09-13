function [anchorbase] = getanchorbase(mode)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
if (mode==0)
    anchorbase=[0,0];
else
if (mode==1)
    anchorbase=[0,0;0,65;65,0;65,65];
else
if (mode==2)
    anchorbase=[0,0;0,65;65,0;65,65;28,0;28,65];
end
end
end
end

