clear all, close all, prwaitbar off;

addpath("../")
addpath("utils/")

dataset = load('datasets/big_dataset.mat');
train = dataset.train;
tst = dataset.tst;

% dataset = load('datasets/small_dataset.mat');
% train = dataset.train_small;
% tst = dataset.tst_small;


%PCA Mapping based on retvar criterion
PCA_mapping = feature_extraction(train, false, 0.95, 0)

%PCA Mapping based on number of classes
%PCA_mapping2 = feature_extraction(train, false, 0, 50)

%PCA + Fishermapping based on retvar for PCA
PCAFISHER_mapping = feature_extraction(train, true, 0.95, 0)

%PCA + Fishermapping based on number of classes
PCAFISHER_mapping2 = feature_extraction(train, true, 0, 30)

%%

% Showing the 5 first components of the pca mapping
figure;
scatterd(train*PCA_mapping(:,1:5), 'gridded');

% Showing the 5 first components of PCA+Fisher (retvar method)
figure
scatterd(train*PCAFISHER_mapping(:,1:5), 'gridded');

% Showing the 5 first components of PCA+Fisher (N method)
figure
scatterd(train*PCAFISHER_mapping2(:,1:5), 'gridded');

%%

%Evaluation of the PCA mapping method
figure
EPCA_NMC = clevalf(train*PCA_mapping, nmc, [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
EPCA_PARZEN = clevalf(train*PCA_mapping, parzenc([],0.5), [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
EPCA_FISHER = clevalf(train*PCA_mapping, fisherc, [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
EPCA_KNN = clevalf(train*PCA_mapping, knnc([], 1), [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
EPCA_LDC = clevalf(train*PCA_mapping, ldc, [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
EPCA_QDC = clevalf(train*PCA_mapping, qdc, [1,2,3,4,5,10:5:100], [], 1, tst*PCA_mapping);
plote({EPCA_NMC, EPCA_PARZEN, EPCA_FISHER, EPCA_KNN, EPCA_LDC, EPCA_QDC});
%%

%Evaluation of the PCA+Fisher mapping
figure
EPCAF = clevalf(train*PCAFISHER_mapping, nmc, 1:9, [],  1, tst*PCAFISHER_mapping);
EPCAF_PARZEN = clevalf(train*PCAFISHER_mapping, parzenc([],0.5), 1:9, [], 1, tst*PCAFISHER_mapping);
EPCAF_FISHER = clevalf(train*PCAFISHER_mapping, fisherc, 1:9, [], 1, tst*PCAFISHER_mapping);
EPCAF_KNN = clevalf(train*PCAFISHER_mapping, knnc([], 1), 1:9, [], 1, tst*PCAFISHER_mapping);
plote({EPCAF, EPCAF_PARZEN, EPCAF_FISHER, EPCAF_KNN});

%%
%Evaluation on PCA+Fisher (reduced N)
figure
EPCAF2 = clevalf(train*PCAFISHER_mapping2, nmc, 1:9, [],  1, tst*PCAFISHER_mapping2);
EPCAF2_PARZEN = clevalf(train*PCAFISHER_mapping2, parzenc([],0.5), 1:9, [], 1, tst*PCAFISHER_mapping2);
EPCAF2_FISHER = clevalf(train*PCAFISHER_mapping2, fisherc, 1:9, [], 1, tst*PCAFISHER_mapping2);
EPCAF2_KNN = clevalf(train*PCAFISHER_mapping2, knnc([], 1), 1:9, [], 1, tst*PCAFISHER_mapping2);
EPCAF2_LDC = clevalf(train*PCAFISHER_mapping2, ldc, 1:9, [], 1, tst*PCAFISHER_mapping2);
EPCAF2_QDC = clevalf(train*PCAFISHER_mapping2, qdc, 1:9, [], 1, tst*PCAFISHER_mapping2);
plote({EPCAF2, EPCAF2_PARZEN, EPCAF2_FISHER, EPCAF2_KNN, EPCAF2_LDC, EPCAF2_QDC});


%% CONCLUSIONS

% CONCLUSION: Pour le BIG Dataset
%    - Globalement, la meilleure méthode est PCA + fisher avec N limité. ça
%    permet d'avoir un minimum d'erreur à seulement 6 features pour à peut
%    près tous les classifiers (un peu moins de 20% d'erreur)

%    - Il y a une exception: 1-NN fonctionne SUPER bien avec le simple PCA,
%    à environ 120 features. L'erreur de classification est inférieure à
%    10%. Par contre dans ce cas là tous les autres classifiers font du
%    caca


% CONCLUSION: Pour le SMALL Dataset
%    Etonnament je n'observe pas de résultats très différents par rapport
%    aux cas précédent