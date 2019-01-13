function error = nist_eval(dataset, classifier)
    
    %This function performs the evaluation of the classifier on a hidden
    %dataset
    
    %Perform classification on unlabelled data
    classified_labels = dataset*classifier*labeld;
    
    %original labels
    labels = getlab(dataset);
    
    %compare -> 0 if error
    success = labels(:,end) - classified_labels(:,end)==0;
    
    
    sz = size(dataset);
    error = 1 - nnz(success)/sz(1);
    
end