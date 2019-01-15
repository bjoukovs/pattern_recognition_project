clear all, close all, prwaitbar off; prwarning off;

dataset = load('datasets/small_dataset.mat');
train = dataset.train;
tst = dataset.tst;

nist_testing = load('datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

%PCA 120 features
PCA_mapping = feature_extraction(train, false, 0.95, 0);

%% TESTING SIMPLE KNN and simple QDC CLASSIFIER
% The aim is to find the best number of features for each of the classifiers
% Full results in featcurve_rep6.mat file
% Best is the neural network combination, bagged qdc, qdc, and bagged 1nn
% number of features respectively: 21, 21, 21, and 26
% errors: 6.8% 7% 7.5% 8.5%

mapped_train = train*knn1_mapping;
mapped_tst = tst*knn1_mapping;
combined_classfr = [knnc([],1)*classc qdc*classc]*bpxnc([],12,5000);
classfr_bagqdc = baggingc([],qdc);
classfr_bagknnc = baggingc([],knnc([], 1));


%Feature curves
repetitions = 6;
feat_number = 1:5:50;
tic
featcurve_1nn = clevalf(mapped_train, knnc([], 1), feat_number, [], repetitions, mapped_tst);
disp('1nn');
featcurve_2nn = clevalf(mapped_train, knnc([], 2), feat_number, [], repetitions, mapped_tst);
disp('2nn');
featcurve_qdc = clevalf(mapped_train, qdc, feat_number, [], repetitions, mapped_tst);
disp('qdc');
featcurve_ldc = clevalf(mapped_train, ldc, feat_number, [], repetitions, mapped_tst);
disp('ldc');
featcruve_combined = clevalf(mapped_train, combined_classfr, feat_number, [], repetitions, mapped_tst);
disp('combined');
featcruve_bagqdc = clevalf(mapped_train, classfr_bagqdc, feat_number, [], repetitions, mapped_tst);
disp('bagqdc');
featcruve_bagknnc = clevalf(mapped_train, classfr_bagknnc, feat_number, [], repetitions, mapped_tst);
disp('bagknnc');
%plote({featcurve_1nn, featcurve_2nn, featcurve_qdc, featcurve_ldc, featcruve_combined, featcruve_bagqdc, featcruve_bagknnc});
toc

% Finding the number of features with the lowest error
[optimalfeat_error_1nn, optimalfeat_index_1nn]=min(featcurve_1nn.error);
optimalfeat_1nn = feat_number(optimalfeat_index_1nn);

[optimalfeat_error_2nn, optimalfeat_index_2nn]=min(featcurve_2nn.error);
optimalfeat_2nn = feat_number(optimalfeat_index_2nn);

[optimalfeat_error_qdc, optimalfeat_index_qdc]=min(featcurve_qdc.error);
optimalfeat_qdc = feat_number(optimalfeat_index_qdc);

[optimalfeat_error_ldc, optimalfeat_index_ldc]=min(featcurve_ldc.error);
optimalfeat_ldc = feat_number(optimalfeat_index_ldc);

[optimalfeat_error_combined, optimalfeat_index_combined]=min(featcruve_combined.error);
optimalfeat_combined = feat_number(optimalfeat_index_combined);

[optimalfeat_error_bagqdc, optimalfeat_index_bagqdc]=min(featcruve_bagqdc.error);
optimalfeat_bagqdc = feat_number(optimalfeat_index_bagqdc);

[optimalfeat_error_bagknnc, optimalfeat_index_bagknnc]=min(featcruve_bagknnc.error);
optimalfeat_bagknnc = feat_number(optimalfeat_index_bagknnc);


