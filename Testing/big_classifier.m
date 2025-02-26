clear all, close all, prwaitbar off;

addpath("../")

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
featcurve1 = clevalf(mapped_train, knn1, 1:2:60, [], 1, mapped_tst);
featcurve2 = clevalf(mapped_train, knn2, 1:2:60, [], 1, mapped_tst);
featcurve3 = clevalf(mapped_train, qdc, 1:2:60, [], 1, mapped_tst);
plote({featcurve1, featcurve2, featcurve3});
 
mapped_train = train*PCA_mapping(:,1:25);
mapped_tst = tst*PCA_mapping(:,1:25);


%train classifier
classfr = knnc(mapped_train, 1);
classfr2 = knnc(mapped_train,2);
classfr3 = qdc(mapped_train);
[E C] = mapped_tst*classfr*testc;
[E2 C2] = mapped_tst*classfr2*testc;
[E3 C3] = mapped_tst*classfr3*testc;

%Confusion matrix
confusion = confmat(getlab(mapped_tst), mapped_tst*classfr*labeld)


%testing on NIST_EVAL
final_classifier = PCA_mapping(:,1:50)*classfr;

error = nist_eval(nist_data, final_classifier);



%% COMBINING KNN + QDC USING 2 LAYER 10-nodes NN


mapped_train = train*PCA_mapping(:,1:25);
mapped_tst = tst*PCA_mapping(:,1:25);

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


featcurve_bp = clevalf(mapped_train, combined_classfr, 1:2:50, [], 1, mapped_tst);
plote(featcurve_bp)


final_classifier = PCA_mapping(:,1:25)*combined_classfr_train;

error = nist_eval(nist_data, final_classifier);

disp('finished');

%Just for curiosity, redoing the feature curves for this classifier:
%featcurve1 = clevalf(train*PCA_mapping, [knnc([],1)*classc qdc*classc]*bpxnc([],10,1000), 20:5:100, [], 1, tst*PCA_mapping);
%plote(featcurve1);
%It seems that no improvements can be made here...


%% Alternative: bagging the qdc classifier

%Just for curiosity, redoing the feature curves for this classifier:
mapped_train = train*PCA_mapping(:,1:50);
mapped_tst = tst*PCA_mapping(:,1:50);
featcurve1 = clevalf(train*PCA_mapping, baggingc([], qdc, 100), 1:2:50, [], 1, tst*PCA_mapping);
plote(featcurve1);
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

%% SVC
% 3.1% error for 30 features
% min error at 21. clevalf takes hours to run on svc(proxm('d',3))*fisherc
% feat_number = 1:5:46;
% featcurve_svc = clevalf(mapped_train, svc(proxm('d',3))*fisherc, feat_number, [], 1, mapped_tst);
% plote(featcurve_svc)

knn1_mapping = PCA_mapping(:,1:30);
mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;

classfr_svc = svc(proxm('d',3))*fisherc; %6.4
classfr_svc_train = knn1_mapping*classfr_svc(mapped_train);
e_svc = nist_data*classfr_svc_train*testc



%% Some ROC curves

mapped_train = train*PCA_mapping(:,1:30);
mapped_tst = tst*PCA_mapping(:,1:30);

%Bagging QDC
bagging_c  = baggingc(mapped_train, qdc, 100);

%Combined
combined_classfr = [knnc([],1)*classc qdc*classc]*bpxnc([],10,5000);
combined_classfr_train = combined_classfr(mapped_train);

%Single
qdcc = qdc(mapped_train);
knn1c = knnc(mapped_train, 1);

E1 = prroc(mapped_tst, bagging_c);
E2 = prroc(mapped_tst, combined_classfr_train);
E3 = prroc(mapped_tst, qdcc);
E4 = prroc(mapped_tst, knn1c);
plote({E1 E2, E3, E4});
