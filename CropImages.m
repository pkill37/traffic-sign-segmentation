clear
%% current folder path
currentFolder = pwd;
% add folder to current path
imagesFolder = strcat(currentFolder, '\STOP');
resizedImagesFolder = strcat(currentFolder, '\STOPImages');

cd(imagesFolder)
% read folder content
Images = dir;
%image number
im = 1;
for i =1:length(Images)
    close all
    if(strcmp(Images(i).name,'.') == 0 && strcmp(Images(i).name,'..') == 0)
        image = imread(Images(i).name);
        
        cd(currentFolder)
        % find posible cropping size 
        [croppingWidth, croppingHeight] = findPossibleCroppingSize(Images(i), imagesFolder);
        
        figure
        imshow(image)
        %mark upper left corner of cropping rectangle (user input)
        [y, x] =ginput(1);
        % crop image
        cropImage = image(x : x + croppingWidth, y : y + croppingHeight, :);
        imshow(cropImage)
        
        %create name for image
        str = strcat(strcat('stop',int2str(im)),'.png');
        
        %resize image
        resizedImage = imresize(cropImage, [224 224]);
        
        %save image on file location 
        cd(resizedImagesFolder)
        imwrite(resizedImage, str)
        cd(imagesFolder)
        
        % increase image number
        im = im + 1;      
    end
end
cd(currentFolder)
