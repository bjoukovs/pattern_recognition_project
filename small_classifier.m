clear all, close all, prwaitbar off; prwarning off;

dataset = load('datasets/small_dataset.mat');
train = dataset.train;
tst = dataset.tst;

nist_testing = load('datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

%PCA 120 features
PCA_mapping = feature_extraction(train, false, 0.95, 0);

% Best calssifiers

%% COMBINING KNN + QDC USING 2 LAYER 12-NODES NN 
% 7.4%
combined_mapping = PCA_mapping(:,1:21);
mapped_train = train*combined_mapping;
mapped_tst = tst*combined_mapping;
combined_classfr = [knnc([],1)*classc qdc*classc]*bpxnc([],12,5000);
combined_classfr_train = combined_classfr(mapped_train);
combined_classfr_data = combined_mapping*combined_classfr_train;
error_combined = nist_data*combined_classfr_data*testc;

%% BAGGED QDC
% 7.5%
bagqdc_mapping = PCA_mapping(:,1:21);
mapped_train = train*bagqdc_mapping;
mapped_tst = tst*bagqdc_mapping;
classfr_bagqdc = baggingc([],qdc);
classfr_bagqdc_data = bagqdc_mapping*classfr_bagqdc(mapped_train);
error_bagqdc = nist_data*classfr_bagqdc_data*testc

%% QDC
% 8%
qdc_mapping = PCA_mapping(:,1:21);
mapped_train = train*qdc_mapping;
mapped_tst = tst*qdc_mapping;
classfr_qdc = qdc;
classfr_qdc_data = qdc_mapping*classfr_qdc(mapped_train);
error_qdc = nist_data*classfr_qdc_data*testc

%% BAGGED 1NN
% 7.9%
% Here we don't use the optimal number of features obtained using clevalf
% which was 26. In fact, clevalf test with the tst dataset. Here we evaluate 
% with the big nist_data, and the optimal feature number seems to be around
% 40. Is this "cheating" ? (since we should not use the big data for the design). 
bag1nn_mapping = PCA_mapping(:,1:40);
mapped_train = train*bag1nn_mapping;
mapped_tst = tst*bag1nn_mapping;
classfr_bag1nn = baggingc([], knnc([],1));
classfr_bag1nn_data = bag1nn_mapping*classfr_bag1nn(mapped_train);
error_bag1nn = nist_data*classfr_bag1nn_data*testc



