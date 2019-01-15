clear all, close all, prwaitbar off;

dataset = load('datasets/small_dataset.mat');
train = dataset.train;
tst = dataset.tst;

nist_testing = load('datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

%PCA 120 features
PCA_mapping = feature_extraction(train, false, 0.95, 0);



%% TESTING SIMPLE KNN and simple QDC CLASSIFIER

mapped_train = train*PCA_mapping;
mapped_tst = tst*PCA_mapping;

knn1 = knnc([], 1);
knn2 = knnc([], 2);

%Feature curves
feat_number = 1:10:150;
featcurve1 = clevalf(mapped_train, knn1, feat_number, [], 1, mapped_tst);
% featcurve2 = clevalf(mapped_train, knn2, feat_number, [], 1, mapped_tst);
% featcurve3 = clevalf(mapped_train, qdc, feat_number, [], 1, mapped_tst);
plote({featcurve1, featcurve2, featcurve3});

[L,optimal_feat_index]=min(featcurve1.error);
optimal_feat = feat_number(optimal_feat_index);
knn1_mapping = PCA_mapping(:,1:optimal_feat);

mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;


%train classifier
classfr = knnc(mapped_train, 1);
[E C] = mapped_tst*classfr*testc;

%testing on NIST_EVAL
final_classifier = knn1_mapping*classfr;
error = nist_eval(nist_data, final_classifier);

%% COMBINING KNN + QDC USING 2 LAYER 12-NODES NN
% Best 

knn1_mapping = PCA_mapping(:,1:21);
mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;

w1 = knnc(mapped_train, 1);
w3 = qdc(mapped_train);

combined_classfr = [knnc([],1)*classc qdc*classc]*bpxnc([],12,5000);
combined_classfr_train = combined_classfr(mapped_train);

[E C] = mapped_tst*combined_classfr_train*testc;


final_classifier = knn1_mapping*combined_classfr_train;

error1 = nist_data*final_classifier*testc;
error_list = [error_list error1];


%% COMBINING KNN + QDC USING DIFFERENT COMBINERS
% results: emin=0.0800    emax=0.0950    eprod=0.0800    emedian=0.0950    evote=0.0850
% best is prodc and minc at 8%

knn1_mapping = PCA_mapping(:,1:21);
mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;

w1 = knnc([], 1);
w3 = qdc([]);

combined_max = [w1 w3]*maxc;
combined_min = [w1 w3]*minc;
combined_votec = [w1 w3]*votec;
combined_medianc = [w1 w3]*medianc;
combined_prodc = [w1 w3]*prodc;
combined_meanc = [w1 w3]*meanc;

% Final combined classifiers
combined_classfr_train_max = knn1_mapping*combined_max(mapped_train);
combined_classfr_train_min = knn1_mapping*combined_min(mapped_train);
combined_classfr_train_votec = knn1_mapping*combined_votec(mapped_train);
combined_classfr_train_medianc = knn1_mapping*combined_medianc(mapped_train);
combined_classfr_train_prodc = knn1_mapping*combined_prodc(mapped_train);
combined_classfr_train_meanc = knn1_mapping*combined_meanc(mapped_train);

% Errors
emean = nist_data*combined_classfr_train_meanc*testc;
emin = nist_data*combined_classfr_train_min*testc
emax = nist_data*combined_classfr_train_max*testc
eprod = nist_data*combined_classfr_train_prodc*testc
emedian = nist_data*combined_classfr_train_medianc*testc
evote = nist_data*combined_classfr_train_votec*testc

%% Decision tree
% Very bad 31%

knn1_mapping = PCA_mapping(:,1:optimal_feat);
mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;

for i=0:1:10
    classfr_tree = treec([],'infcrit',i);
    classfr_train_tree = knn1_mapping*classfr_tree(mapped_train);
    e_tree = nist_data*classfr_train_tree*testc
end


%% Bagging kcc, qdc, and ldc
% Best is bagging qdc

knn1_mapping = PCA_mapping(:,1:21);
mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;

classfr_knnbag = baggingc([],knnc([],1));
classfr_knnbag_train = knn1_mapping*classfr_knnbag(mapped_train);
e_knnbag = nist_data*classfr_knnbag_train*testc

% for qdc 0.0850
% for knnc 0.0900
% for ldc 0.1130

%% AdaBoosting 

% knn1_mapping = PCA_mapping(:,1:optimal_feat);
% mapped_train = train*knn1_mapping;
% mapped_tst = tst*knn1_mapping;
% 
% for i = [1,10,100,1000]
%     classfr_ada = adaboostc([],qdc,i,[],0);
%     classfr_ada_train = knn1_mapping*classfr_ada(mapped_train);
%     e_ada = nist_data*classfr_ada_train*testc
% 
% end


%% COMBINING BAGGING QDC AND BAGGING KNN1
% Same/ slightly worse than bagged qdc and knn1

mapped_train = train*PCA_mapping(:,1:21);
mapped_tst = tst*PCA_mapping(:,1:21);

bagging_qdc = baggingc([], qdc, 100);
bagging_1nn = baggingc([], knnc([],1), 10);
combined = [bagging_qdc*classc bagging_1nn*classc]*maxc;

trained_combined = PCA_mapping(:,1:21)*combined(mapped_train);
e_knnqdc = nist_data*trained_combined*testc




