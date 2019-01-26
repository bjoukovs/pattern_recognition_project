clear all, close all, prwaitbar off; prwarning off;

struct = load('../datasets/small_dataset_notfolded.mat');
dataset = struct.DATA1;
nist_testing = load('../datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

PCA_mapping = feature_extraction(dataset, false, 0.95, 0);

%% ldc 
errldc_list = [];
feat_number = 5:5:40;
repetitions = 1;
for i=feat_number
    test_map_ldc = PCA_mapping(:,1:i);
    mapped_train = dataset*test_map_ldc;
    [ERR,CERR,NLAB_OUT] = prcrossval(mapped_train,ldc);
    errldc_list = [errldc_list ERR];
end

[optimalfeat_error_ldc, optimalfeat_index_ldc]=min(errldc_list);
optimalfeat_ldc = feat_number(optimalfeat_index_ldc);

mapped_train = dataset*PCA_mapping(:,1:optimalfeat_ldc);
classfr_ldc = PCA_mapping(:,1:optimalfeat_ldc)*ldc(mapped_train);
error_ldc = nist_data*classfr_ldc*testc;

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

