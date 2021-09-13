function [flag] = writexml(filename,BB,label,mode,num_in_one)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明

if (mode==0)
    % create document
    docNode = com.mathworks.xml.XMLUtils.createDocument('Annotation');

    % document element
    docRootNode = docNode.getDocumentElement();

    % folder
    folderNode = docNode.createElement('folder');
    folderNode.appendChild(docNode.createTextNode(sprintf('train_images')));
    docRootNode.appendChild(folderNode);

    % filename
    filenameNode = docNode.createElement('filename');
    filenameNode.appendChild(docNode.createTextNode(sprintf([filename,'.jpg'])));
    docRootNode.appendChild(filenameNode);

    % size
    sizeElement = docNode.createElement('size'); 
    docRootNode.appendChild(sizeElement);

    widthNode = docNode.createElement('width');
    widthNode.appendChild(docNode.createTextNode('64'));
    sizeElement.appendChild(widthNode);

    heightNode = docNode.createElement('height');
    heightNode.appendChild(docNode.createTextNode('64'));
    sizeElement.appendChild(heightNode);

    depthNode = docNode.createElement('depth');
    depthNode.appendChild(docNode.createTextNode('1'));
    sizeElement.appendChild(depthNode);

    % object
    objectElement = docNode.createElement('object'); 
    docRootNode.appendChild(objectElement);

    nameNode = docNode.createElement('name');
    nameNode.appendChild(docNode.createTextNode(label));
    objectElement.appendChild(nameNode);

    difficultNode = docNode.createElement('difficult');
    difficultNode.appendChild(docNode.createTextNode('0'));
    objectElement.appendChild(difficultNode);

    %object-bndbox
    bndboxElement = docNode.createElement('bndbox'); 
    objectElement.appendChild(bndboxElement);

    xminNode = docNode.createElement('xmin');
    xminNode.appendChild(docNode.createTextNode(num2str(BB(1))));
    bndboxElement.appendChild(xminNode);

    yminNode = docNode.createElement('ymin');
    yminNode.appendChild(docNode.createTextNode(num2str(BB(2))));
    bndboxElement.appendChild(yminNode);

    xmaxNode = docNode.createElement('xmax');
    xmaxNode.appendChild(docNode.createTextNode(num2str(BB(3))));
    bndboxElement.appendChild(xmaxNode);

    ymaxNode = docNode.createElement('ymax');
    ymaxNode.appendChild(docNode.createTextNode(num2str(BB(4))));
    bndboxElement.appendChild(ymaxNode);
    % xmlwrite
    xmlFileName = ['D:\seminar\selfmadelabel',filename,'.xml'];
    xmlwrite(xmlFileName,docNode);
else
    % create document
    docNode = com.mathworks.xml.XMLUtils.createDocument('Annotation');

    % document element
    docRootNode = docNode.getDocumentElement();

    % folder
    folderNode = docNode.createElement('folder');
    folderNode.appendChild(docNode.createTextNode(sprintf('train_images')));
    docRootNode.appendChild(folderNode);

    % filename
    filenameNode = docNode.createElement('filename');
    filenameNode.appendChild(docNode.createTextNode(sprintf([filename,'.jpg'])));
    docRootNode.appendChild(filenameNode);

    % size
    sizeElement = docNode.createElement('size'); 
    docRootNode.appendChild(sizeElement);

    widthNode = docNode.createElement('width');
    widthNode.appendChild(docNode.createTextNode('64'));
    sizeElement.appendChild(widthNode);

    heightNode = docNode.createElement('height');
    heightNode.appendChild(docNode.createTextNode('64'));
    sizeElement.appendChild(heightNode);

    depthNode = docNode.createElement('depth');
    depthNode.appendChild(docNode.createTextNode('1'));
    sizeElement.appendChild(depthNode);
    
    for i=1:num_in_one
        % object
        objectElement = docNode.createElement('object'); 
        docRootNode.appendChild(objectElement);

        nameNode = docNode.createElement('name');
        nameNode.appendChild(docNode.createTextNode(label{i}));
        objectElement.appendChild(nameNode);

        difficultNode = docNode.createElement('difficult');
        difficultNode.appendChild(docNode.createTextNode('0'));
        objectElement.appendChild(difficultNode);

        %object-bndbox
        bndboxElement = docNode.createElement('bndbox'); 
        objectElement.appendChild(bndboxElement);

        xminNode = docNode.createElement('xmin');
        xminNode.appendChild(docNode.createTextNode(num2str(BB{i}(1))));
        bndboxElement.appendChild(xminNode);

        yminNode = docNode.createElement('ymin');
        yminNode.appendChild(docNode.createTextNode(num2str(BB{i}(2))));
        bndboxElement.appendChild(yminNode);

        xmaxNode = docNode.createElement('xmax');
        xmaxNode.appendChild(docNode.createTextNode(num2str(BB{i}(3))));
        bndboxElement.appendChild(xmaxNode);

        ymaxNode = docNode.createElement('ymax');
        ymaxNode.appendChild(docNode.createTextNode(num2str(BB{i}(4))));
        bndboxElement.appendChild(ymaxNode);
    end
    
    % xmlwrite
    xmlFileName = ['D:\seminar\selfmadedata-180Kpng\selfmadelabel\',filename,'.xml'];
    xmlwrite(xmlFileName,docNode);

flag=0;
end

