function [flag] = Formtrainingset(numofpic,mode)
%randseed=300
%numofpic is the total number of pictures in the trainingset
%mode=0 -> 1 num per img
%mode=1 -> 3-4 num per img
%mode=2 -> 5-6 num per img
%e.g. Formtrainingset(60000,1)
tic
rng(300);
Mylist=ls('train_images');%The original mnist dataset
Mylist=Mylist(3:end,:);
BB=cell(1,numofpic);
for i=1:numofpic
    pic=imread(['train_images\',Mylist(i,:)]);
    BB{i}=getboundingbox(pic);
end 
if(mode==0)
    for i=1:numofpic
        num_in_one=1;
        No_of_img=randi(size(Mylist,1),1,num_in_one);
        packofimg=cell(1,num_in_one);
        packofBB=cell(1,num_in_one);
        for j=1:num_in_one
            packofimg{j}=['train_images\',Mylist(No_of_img(j),:)];
            packofBB{j}=BB{No_of_img(j)};
        end
        Constructdataset(packofimg,i,0,num_in_one,packofBB);
    end
end
if(mode==1)
    for i=1:numofpic
        num_in_one=randi([3,4]);
        No_of_img=randi(size(Mylist,1),1,num_in_one);
        packofimg=cell(1,num_in_one);
        packofBB=cell(1,num_in_one);
        for j=1:num_in_one
            packofimg{j}=['train_images\',Mylist(No_of_img(j),:)];
            packofBB{j}=BB{No_of_img(j)};
        end
        Constructdataset(packofimg,i,1,num_in_one,packofBB);
    end
end
if(mode==2)
    for i=1:numofpic
        num_in_one=randi([5,6]);
        No_of_img=randi(size(Mylist,1),1,num_in_one);
        packofimg=cell(1,num_in_one);
        packofBB=cell(1,num_in_one);
        for j=1:num_in_one
            packofimg{j}=['train_images\',Mylist(No_of_img(j),:)];
            packofBB{j}=BB{No_of_img(j)};
        end
        Constructdataset(packofimg,i,2,num_in_one,packofBB);
    end
end
flag=0;
toc
end


