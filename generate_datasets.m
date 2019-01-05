clear all, close all, prwaitbar off


% The NIST datasets contains 1000 images per classes
% This function generates the various datasets for the classifier design
% and nist_eval

datasets = my_rep(prnist([0:9], [1:1000]));


%Evaluate classifierd on hidden data
%error1 = nist_eval(datasets(2), big_classifier);
%error2 = nist_eval(datasets(4), big_classifier);