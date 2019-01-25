function classfr = gen_classifier_big(train, tst)
    
    % Generates the digit classifier including the feature selection
    % mapping, in the case of the small dataset
    
    % The chosen classifier is 3rd order polynomial kernelized SVC + fisher
    
    % PCA retvar = 95%
    PCA_mapping = feature_extraction(train, false, 0.95, 0);
    
    %Mapping training sets with 21 features only
    svc_mapping = PCA_mapping(:,1:21);
    mapped_train = train*svc_mapping;
    mapped_tst = tst*svc_mapping;
    
    %Training classifier
    classfr_svc = svc(proxm('d',3))*fisherc;  
    classfr_svc_train = classfr_svc(mapped_train);
    
    classfr = svc_mapping*classfr_svc_train;
    

end

