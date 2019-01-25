function classfr = gen_classifier_big(train, tst)
    
    % Generates the digit classifier including the feature selection
    % mapping, in the case of the big dataset
    
    % The chosen classifier is the bagged QDC
    
    % PCA retvar = 95%
    PCA_mapping = feature_extraction(train, false, 0.95, 0);
    
    % mapped training and test sets on 30 features
    mapped_train = train*PCA_mapping(:,1:30);
    mapped_tst = tst*PCA_mapping(:,1:30);
    
    % Bagging QDC 100
    bagging_c  = baggingc(mapped_train, qdc, 100);
    
    classfr = PCA_mapping(:,1:30)*bagging_c;

end

