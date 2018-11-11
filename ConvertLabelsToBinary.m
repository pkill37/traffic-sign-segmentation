clear
folder = pwd;
% add label folder name
labels = strcat(folder,'\STOPLabels');

cd(labels)
colorLabels = dir;

for j= 1:length(colorLabels)
     close
     if(strcmp(colorLabels(j).name,'.') == 0 && strcmp(colorLabels(j).name,'..') == 0)
         imageLabel = (im2double(rgb2gray(imread(colorLabels(j).name))));
         sizeI = size(imageLabel);
                binary = imageLabel;
                for k=1:sizeI(1)
                    for l=1:sizeI(2)
                        if(binary(k,l) ~= 0)
                            binary(k,l) = 1;
                        else
                            binary(k,l) = 0;
                        end
                    end
                end
                figure
                imshow(binary)
                % this should overwrite existing label
                imwrite(binary, colorLabels(j).name)
     end
 end