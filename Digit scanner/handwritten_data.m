
img_files = dir('handwritten_extracted/*.jpg');
img = imread(strcat(img_files(i).folder,'/',img_files(i).name))
dataset({size()}, 1);

% n_imgs = length(img_files)
% imgs = {};
% for i = 1:n_imgs
%     imgs = [imgs; imread(strcat(img_files(i).folder,'/',img_files(i).name))];
% end
% p = imgs(1,1)
% imshow(p)

% dataset({size(im2double(im4(:, :, 1) < 100))}, 1);

% img = imread('handwritten_digits_only.jpg');
% [hight, width, numberOfColorChannels] = size(img);
% img = rgb2gray(img);
% img = img(:,:)<210;
% img = imclose(img, [1 1 1; 1 1 1; 1 1 1;]);
% 
% % use histogram method to detect boundaries
% horizontal_histogram = sum(img, 1) > 0;
% vertical_histogram = sum(img, 1) > 0;
% df = [-1 1];
% horizontal_sep = conv(horizontal_histogram,df,'valid');
% vertical_sep = conv(vertical_histogram,df,'valid');
% for y=1:length(vertical_sep)
%     if horizontal
%     for i=1:length(horizontal_sep)
%         if horizontal_sep(i) == -1
%             line([i,i],[0,hight])
%         end
%     end
% end
% imshow(img)