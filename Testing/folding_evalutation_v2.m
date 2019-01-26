clear all, close all, prwaitbar on; prwarning off;

addpath("../")

struct = load('datasets/small_dataset_notfolded.mat');
dataset = struct.DATA1;
nist_testing = load('datasets/small_nist_eval.mat');
nist_data = nist_testing.DATA2;

% Getting features with im_measure
data1 = im_features(dataset)

%% Mapping using feature selection

[mapping, r] = featself(data1, 'eucl-m')

permutations = randperm(100);
data1_shuff = data1(permutations, :)

%% TESTING VARIOUS CLASSIFIERS : USING 10 repetitions
feat_number = 2:1:20;

figure
EPCA_NMC = clevalf(data1*mapping, nmc, feat_number, [], 10);
EPCA_PARZEN = clevalf(data1*mapping, parzenc([],0.5), feat_number, [], 10);
EPCA_FISHER = clevalf(data1*mapping, fisherc, feat_number, [], 10);
EPCA_KNN = clevalf(data1*mapping, knnc([], 1), feat_number, [], 10);
EPCA_LDC = clevalf(data1*mapping, ldc, feat_number, [], 10);
EPCA_QDC = clevalf(data1*mapping, qdc, feat_number, [], 10);
EPCA_SVC = clevalf(dataset*PCA_mapping, svc(proxm('d',3))*fisherc, feat_number, [], 2);
plote({EPCA_NMC, EPCA_PARZEN, EPCA_FISHER, EPCA_KNN, EPCA_LDC, EPCA_QDC, EPCA_SVC});
%plote({EPCA_NMC, EPCA_PARZEN, EPCA_FISHER, EPCA_KNN, EPCA_LDC, EPCA_QDC});

%% Fisher and LDC (no mapping used since the more features the best)

[E1 C1 NLABOUT] = prcrossval(data1_shuff, fisherc, 100); %0.28
[E2 C2 NLABOUT] = prcrossval(data1_shuff, ldc, 100);     %0.21
[E3 C3 NLABOUT] = prcrossval(data1_shuff, baggingc([], ldc, 100), 10);   %0.25
[E4 C4 NLABOUT] = prcrossval(data1_shuff, baggingc([], fisherc, 100), 10); %0.31
[E5 C5 NLABOUT] = prcrossval(data1_shuff, adaboostc([], ldc, 100), 10); %0.29

%% Evaluating final classifiers

%LDC
ldcc = ldc(data1_shuff);
ldcc_train = im_features([])*ldcc;

errors_ldc = nist_eval(nist_data, ldcc_train, 300)
