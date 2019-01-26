function error = nist_eval(dataset, classifier, n)
    
    global CHECK_SIZES;
    CHECK_SIZES = false;
    
    %This function performs the evaluation of the classifier on a hidden
    %dataset, for n random digits
    
    %get random indices
    sz = size(dataset);
    idx = randperm(sz(1));
    idx = idx(1:n);
    
    sub_dataset = dataset(idx,:);
    
    %Perform classification on unlabelled data
    classified_labels = sub_dataset*classifier*labeld;
    
    %original labels
    labels = getlab(sub_dataset);
    
    %compare -> 0 if error
    success = labels(:,end) - classified_labels(:,end)==0;
    
    error = 1 - nnz(success)/n;
    
end