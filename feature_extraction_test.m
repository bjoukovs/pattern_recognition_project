clear all, close all, prwaitbar off;

dataset = load('datasets/big_dataset.mat');
train = dataset.train_big;
tst = dataset.tst_big;

%PCA Mapping based on retvar criterion
PCA_mapping = feature_extraction(train, false, 0.95, 0)

%PCA Mapping based on number of classes
PCA_mapping2 = feature_extraction(train, false, 0, 50)

%PCA + Fishermapping based on retvar for PCA
PCAFISHER_mapping = feature_extraction(train, true, 0.95, 0)

%PCA + Fishermapping based on number of classes
PCAFISHER_mapping = feature_extraction(train, true, 0, 50)