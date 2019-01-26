function datasets = my_rep(datafile)

    % This function generates datasets for the large dataset case and small
    % dataset case.
    % The output is a cell of datasets containing
    %   1) Big dataset for classifier training and testing
    %   2) Big dataset for classifier evaluation (nist_eval)
    %   3) Small dataset for classifier training and testing
    %   7) Small dataset for classifier evaluation (nist_eval)
    
    datasets = {'../datasets/big_nist_eval.mat', '../datasets/big_dataset.mat', '../datasets/small_nist_eval.mat', '../datasets/small_dataset.mat', '../datasets/small_dataset_notfolded.mat'};
    
    %Step 1 : Manipulate the images, resize them, make them square...
    
    %Resizing and boxing images to 64x64
    data = datasetm(im_resize(im_box(datafile, 0, 1), [64 64]));

    
    %Step 3 : Making the big and small datasets
    %Case 1: Big training set

    [DATA1, DATA2] = gendat(data,0.5);

    save((datasets{1}), 'DATA2');

    frac = 0.5; %250 images for training, 250 for testing
    [train, tst] = gendat(DATA1, frac);

    save((datasets{2}), 'train', 'tst');




    %Case 2: Small training set

    [DATA1, DATA2] = gendat(data,0.01);

    save(datasets{3}, 'DATA2');

    frac = 0.3; %10 images for training, 90 for testing
    [train, tst] = gendat(DATA1, frac);

    save(datasets{4}, 'train', 'tst');
    save(datasets{5}, 'DATA1');

end



