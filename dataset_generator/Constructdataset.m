function [flag] = Constructdataset(filename,num,mode,num_in_one,BB)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
flag=0;

label=cell(num_in_one);
scale=getscale(mode,num_in_one);
presplice=zeros(64);

anchorbase=getanchorbase(mode);%[0,0;0,65;65,0;65,65];
anchorsign=getanchorsign(mode);%[1,1;1,-1;-1,1;-1,-1];
delno=randi(2*mode+2);
anchorbase=randdel(anchorbase,mode,num_in_one,delno);
anchorsign=randdel(anchorsign,mode,num_in_one,delno);

for i=1:num_in_one
    %get label from filename
    manylabelstr=strsplit(filename{i},'_');
    labelstr=manylabelstr{3};
    label{i}=labelstr(1);

    %get bounding box
    pic=imread(filename{i});
    pic=pic(BB{i}(2):BB{i}(4),BB{i}(1):BB{i}(3));
    BB{i}=BB{i}-[BB{i}(1)-1,BB{i}(2)-1,BB{i}(1)-1,BB{i}(2)-1];

    %resize and binarize images
    pic=imbinarize(imresize(pic,scale(i)),127/255);

    %getanchor
    anchorbase(i,:)=anchorbase(i,:)+(0.5*anchorsign(i,:)-0.5).*[size(pic,2),size(pic,1)];
    anchor=getrandanchor(64,pic,mode);
    anchor=anchorbase(i,:)+anchorsign(i,:).*anchor;

    %splice images
    union=splice(64,pic,BB{i},scale(i),presplice,anchor);
    presplice=union{1};
    BB{i}=union{2};
end

%testBB(presplice,BB,num_in_one,num,mode);

writexml([num2str(mode),'\',num2str(mode*10000000+num,'%08d')],BB,label,1,num_in_one);%.xml annotations saving path
imwrite(presplice,['D:\seminar\selfmadedata-180Kpng\selfmadeimg\',num2str(mode),'\',num2str(mode*10000000+num,'%08d'),'.png']);%.png imgs saving path        
end


