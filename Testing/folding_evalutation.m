clear all, close all, prwaitbar off; prwarning off;

addpath("../")

struct = load('datasets/small_dataset_notfolded.mat');
dataset = struct.DATA1;
nist_testing = load('datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

PCA_mapping = feature_extraction(dataset, false, 0.95, 0);

%% TESTING VARIOUS CLASSIFIERS : USING 10 repetitions
feat_number = 2:1:20;

figure
EPCA_NMC = clevalf(dataset*PCA_mapping, nmc, feat_number, [], 10);
EPCA_PARZEN = clevalf(dataset*PCA_mapping, parzenc([],0.5), feat_number, [], 10);
EPCA_FISHER = clevalf(dataset*PCA_mapping, fisherc, feat_number, [], 10);
EPCA_KNN = clevalf(dataset*PCA_mapping, knnc([], 1), feat_number, [], 10);
EPCA_LDC = clevalf(dataset*PCA_mapping, ldc, feat_number, [], 10);
EPCA_QDC = clevalf(dataset*PCA_mapping, qdc, feat_number, [], 10);
%EPCA_SVC = clevalf(dataset*PCA_mapping, svc(proxm('d',3))*fisherc, feat_number, [], 10);
%plote({EPCA_NMC, EPCA_PARZEN, EPCA_FISHER, EPCA_KNN, EPCA_LDC, EPCA_QDC, EPCA_SVC});
plote({EPCA_NMC, EPCA_PARZEN, EPCA_FISHER, EPCA_KNN, EPCA_LDC, EPCA_QDC});

%% PARZEN

mapped_data0 = dataset*PCA_mapping(:,1:10);

permutations = randperm(100);
mapped_data = mapped_data0(permutations, :)

parzen1 = parzenc([], 0.5)
parzen2 = parzenc([], 1)
parzen3 = parzenc([], 1.5)

[E1 C1 NLABOUT] = prcrossval(mapped_data, parzen1);
[E2 C2 NLABOUT] = prcrossval(mapped_data, parzen2);
[E3 C3 NLABOUT] = prcrossval(mapped_data, parzen3);

[E4 C4 NLABOUT] = prcrossval(mapped_data, svc)
[E5 C5 NLABOUT] = prcrossval(mapped_data, svc([], proxm('d',3)))

[E6 C6 NLABOUT] = prcrossval(mapped_data, ldc)

[E7 C7 NLABOUT] = prcrossval(mapped_data, treec([], 'infcrit', 10))

[E8 C8 NLABOUT] = prcrossval(mapped_data, baggingc([], ldc, 100), 10)



%% Evaluating final classifiers

mapped_data0 = dataset;

permutations = randperm(100);
mapped_data = mapped_data0(permutations, :)

%LDC bagged
bag_ldc = PCA_mapping(:,1:20)*baggingc([], ldc, 100);
bag_ldc_train = bag_ldc(mapped_data);

[errors_ldc C1] = nist_data*bag_ldc_train*testc

%LDC
ldcc = PCA_mapping(:,1:20)*ldc;
ldcc_train = ldcc(mapped_data);

[errors_ldc2 C2] = nist_data*ldcc_train*testc



%% ldc 
errldc_list = [];
feat_number = 5:1:25;
repetitions = 1;
for i=feat_number
    i
    test_map_ldc = PCA_mapping(:,1:i);
    mapped_train = dataset*test_map_ldc;
    [ERR,CERR,NLAB_OUT] = prcrossval(mapped_train, ldc);
    errldc_list = [errldc_list ERR];
end

[optimalfeat_error_ldc, optimalfeat_index_ldc]=min(errldc_list);
optimalfeat_ldc = feat_number(optimalfeat_index_ldc);

mapped_train = dataset*PCA_mapping(:,1:optimalfeat_ldc);
classfr_ldc = PCA_mapping(:,1:optimalfeat_ldc)*ldc(mapped_train);
error_ldc = nist_data*classfr_ldc*testc;

plot(feat_number, errldc_list);

%% SVC
% 6.4%

errsvc_list = [];
feat_number = 5:5:30;
repetitions = 1;
svc_classi = knnc([], 1);  % replace with the classfier to be tested
for i=feat_number
    test_map_svc = PCA_mapping(:,1:i);
    mapped_train = dataset*test_map_svc;
    [ERR,CERR,NLAB_OUT] = prcrossval(mapped_train,svc_classi);
    errsvc_list = [errsvc_list ERR]
end

[optimalfeat_error_svc, optimalfeat_index_svc]=min(errsvc_list);
optimalfeat_svc = feat_number(optimalfeat_index_svc);

mapped_train = dataset*PCA_mapping(:,1:optimalfeat_svc);
classfr_svc = PCA_mapping(:,1:optimalfeat_svc)*svc_classi(mapped_train);
error_svc = nist_data*classfr_svc*testc

