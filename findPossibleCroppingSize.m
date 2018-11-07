function [croppingWidth, croppingHeight] = findPossibleCroppingSize(image, folder)
cd(folder)
imageInfo = imfinfo(image.name);
minWidth = imageInfo.Width;
minHeight = imageInfo.Height;

minSize = min(minWidth, minHeight);
% set cropping width, and height to min size, but leave some space 
croppingWidth = minSize - 100;
croppingHeight = minSize - 100;


