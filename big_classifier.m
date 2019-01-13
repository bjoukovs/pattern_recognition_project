clear all, close all, prwaitbar off;

dataset = load('datasets/big_dataset.mat');
train = dataset.train;
tst = dataset.tst;

nist_testing = load('datasets/big_nist_eval.mat');
nist_data = nist_testing.DATA2;

%PCA 120 features
PCA_mapping = feature_extraction(train, false, 0.95, 0);



%% TESTING SIMPLE KNN and simple QDC CLASSIFIER

mapped_train = train*PCA_mapping;
mapped_tst = tst*PCA_mapping;

%Normal 1NN classifier
knn1 = knnc([], 1);

%Normal 2NN classifier
knn2 = knnc([], 2);

%Feature curves
featcurve1 = clevalf(mapped_train, knn1, 1:10:150, [], 1, mapped_tst);
featcurve2 = clevalf(mapped_train, knn2, 1:10:150, [], 1, mapped_tst);
featcurve3 = clevalf(mapped_train, qdc, 1:10:150, [], 1, mapped_tst);
plote({featcurve1, featcurve2, featcurve3});
 
mapped_train = train*PCA_mapping(:,1:50);
mapped_tst = tst*PCA_mapping(:,1:50);


%train classifier
classfr = knnc(mapped_train, 1);
%classfr = parzenc(mapped_train,1);
%classfr = qdc(mapped_train);
[E C] = mapped_tst*classfr*testc;

%Confusion matrix
confusion = confmat(getlab(mapped_tst), mapped_tst*classfr*labeld)


%testing on NIST_EVAL
final_classifier = PCA_mapping(:,1:50)*classfr;

error = nist_eval(nist_data, final_classifier);



%% COMBINING KNN + QDC USING 2 LAYER 10-nodes NN


mapped_train = train*PCA_mapping(:,1:50);
mapped_tst = tst*PCA_mapping(:,1:50);

w1 = knnc(mapped_train, 1);
w3 = qdc(mapped_train);

% combined1 = [w1 w3]*maxc;
% combined2 = [w1 w3]*meanc;
% combined3 = [w1 w3]*votec;

[E1 C1] = mapped_tst*w1*testc;
[E3 C3] = mapped_tst*w3*testc;

combined_classfr = [knnc([],1)*classc qdc*classc]*bpxnc([],10,5000);
combined_classfr_train = combined_classfr(mapped_train);

[E C] = mapped_tst*combined_classfr_train*testc;


final_classifier = PCA_mapping(:,1:50)*combined_classfr_train;

error = nist_eval(nist_data, final_classifier);

disp('finished');

%Just for curiosity, redoing the feature curves for this classifier:
%featcurve1 = clevalf(train*PCA_mapping, [knnc([],1)*classc qdc*classc]*bpxnc([],10,1000), 20:5:100, [], 1, tst*PCA_mapping);
%plote(featcurve1);
%It seems that no improvements can be made here...


%% Alternative: bagging the qdc classifier

%Just for curiosity, redoing the feature curves for this classifier:
%featcurve1 = clevalf(train*PCA_mapping, baggingc([], qdc, 100), 20:5:100, [], 1, tst*PCA_mapping);
%plote(featcurve1);
% It seems that we could do even better with only ~35 features so this is
% the reason I put 30 instead of 50 in this case
%The classifier as a super good performance (3.5 %) !

mapped_train = train*PCA_mapping(:,1:30);
mapped_tst = tst*PCA_mapping(:,1:30);

bagging_c  = baggingc(mapped_train, qdc, 100);
[E C] = mapped_tst*bagging_c*testc;

final_classifier = PCA_mapping(:,1:30)*bagging_c;
error = nist_eval(nist_data, final_classifier);
disp('finished');


%% Final test: combining bagging QDC and bagging knn1

% I also tried with bpxnc for combining, does not improve previous result...

% mapped_train = train*PCA_mapping(:,1:30);
% mapped_tst = tst*PCA_mapping(:,1:30);
% 
% bagging_qdc = baggingc([], qdc, 100);
% bagging_1nn = baggingc([], knnc([],1), 10);
% combined = [bagging_qdc*classc bagging_1nn*classc]*maxc;
% 
% trained_combined = mapped_train*combined;

[E C] = mapped_tst*trained_combined*testc;
