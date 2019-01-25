function out = cleanUp(im)
se = strel('disk',1);
im = imclose(im,se);
im = imerode(im,se);
im = imdilate(im,se);
im = slant_correction(im);
level = graythresh(im);
im = im2bw(im,level);
im = imresize(im,[32,32]);
out = im;
end
