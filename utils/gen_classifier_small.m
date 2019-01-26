function [classfr, test_error] = gen_classifier_big(dataset)
    
    % Generates the digit classifier including the feature selection
    % mapping, in the case of the small dataset
    
    % The classifier is trained using all the dataset and the error is
    % estimated using cross-validation
    
    % The chosen classifier is LDC on im_features (all features)
    sz = size(dataset);
    permutations = randperm(sz(1));
    data1_shuff = dataset(permutations, :)

    classifier = ldc(im_features(data1_shuff));
    classfr = im_features([])*classifier;
    
    [E2 C2 NLABOUT] = prcrossval(dataset, im_features([])*ldc, 10);
    test_error = E2;
    

end

