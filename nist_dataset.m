clear all, close all, prwaitbar off;

% The NIST datasets contains 1000 images per classes
NIST = prnist([0:9], [1:1000]);

images_per_class = {};

classes = 1:10;
labels = 0:9;

for i=classes
    
    fprintf('Extracting images for class %d \n', i);
    images_per_class{i} = data2im(seldat(NIST,i));
    
end

save('IMAGES.mat', 'images_per_class');

%%

%Step 1 : Manipulate the images, resize them, make them square...




%Step 2 : making a new dataset using the vectorized images and prdataset()



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

frac = 0.1 %10 images for training, 90 for testing
[train_small, tst_small] = gendat(DATA1, frac);

save("datasets/small_dataset.mat", 'train_small', 'tst_small');

