function error = nist_eval(dataset, classifier)
    
    %This function performs the evaluation of the classifier on a hidden
    %dataset
    
    %Perform classification on unlabelled data
    classified_labels = getlab(+dataset*classifier);
    
    %original labels
    labels = getlab(dataset);
    
    %compare
    success = (classified_labels - labels)==0;
    
    error = nnz(success);
    
end