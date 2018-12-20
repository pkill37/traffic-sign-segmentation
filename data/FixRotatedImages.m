clear
%% current folder path
currentFolder = pwd;
% add folder to current path
imagesFolder = strcat(currentFolder, '/fix');
resizedImagesFolder = strcat(currentFolder, '/fixed');

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

        %create name for image
        str = Images(i).name;

        %resize image
        resizedImage = imrotate(image, -90);

        %save image on file location
        cd(resizedImagesFolder)
        imwrite(resizedImage, str)
        cd(imagesFolder)

        % increase image number
        im = im + 1;
    end
end
cd(currentFolder)
