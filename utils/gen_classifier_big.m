function [classfr, test_error] = gen_classifier_big(train, tst)
    
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
    
    % SVC with d^3 kernel 
%     svc_c = svc(proxm('d',3))*fisherc;
%     svc_c = svc_c(mapped_train);
    
    % Neural network combination of QDC and KN1
    % features must be changed to 24 instead of 30
%     combi_c = [knnc([],1)*classc qdc*classc]*bpxnc([],10,5000);
%     combi_c = combi_c(mapped_train);
    
    classfr = PCA_mapping(:,1:30)*bagging_c;
    
    [E C] = tst*classfr*testc;
    test_error = E;

end

