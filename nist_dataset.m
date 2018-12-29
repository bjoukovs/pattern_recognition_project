clear all, close all, prwaitbar off;

% The NIST datasets contains 1000 images per classes
% Puts them in a box and resize it to 64x64
% The datasetm command needs equal size images to work

NIST = datasetm(im_resize(im_box(prnist([0:9], [1:1000]), 0, 1), [64 64]));


%Step 1 : Manipulate the images, resize them, make them square...



%Step 3 : Making the big and small datasets
%Case 1: Big training set

[DATA1, DATA2] = gendat(NIST,0.5);

save("datasets/big_nist_eval.mat", 'DATA2');

frac = 0.5; %250 images for training, 250 for testing
[train_big, tst_big] = gendat(DATA1, frac);

save("datasets/big_dataset.mat", 'train_big', 'tst_big');




%Case 2: Small training set

[DATA1, DATA2] = gendat(NIST,0.9);

save("datasets/small_nist_eval.mat", 'DATA2');

frac = 0.1; %10 images for training, 90 for testing
[train_small, tst_small] = gendat(DATA1, frac);

save("datasets/small_dataset.mat", 'train_small', 'tst_small');

