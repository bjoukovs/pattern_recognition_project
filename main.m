clear all, close all, prwaitbar on

addpath('utils/');
addpath('Digit_scanner/')

%% Generating the datasets

% The NIST datasets contains 1000 images per classes
% This function generates the various datasets for the classifier design
% and nist_eval

% datasets = my_rep(prnist([0:9], [1:1000]));

% If already generated before, we can simply load it

% For the large dataset case :
big_dataset = load('datasets/big_dataset.mat');
train_big = big_dataset.train;
tst_big = big_dataset.tst;

nist_testing_big = load('datasets/big_nist_eval.mat');
nist_data_big = nist_testing_big.DATA2;

%For the small dataset case
dataset_small = load('datasets/small_dataset_notfolded.mat');
train_small = dataset_small.DATA1;

nist_testing_small = load('datasets/small_nist_eval.mat');
nist_data_small = nist_testing_small.DATA2;


%% Training the big classifier

[classifier_big, error_big] = gen_classifier_big(train_big, tst_big);



%% Training the small classifier

[classifier_small, error_small] = gen_classifier_small(train_small);


%% Testing the classifiers with NIST_EVAL

error_big_nist = nist_eval(nist_data_big, classifier_big, 300);
error_small_nist = nist_eval(nist_data_small, classifier_small, 300);

if error_big_nist < 0.05
    disp(sprintf('Success! The BIG classifier performs with a NIST error of %f', error_big_nist));
else
    disp(sprintf('The BIG classifier does not fulfill the requirements: NIST error of %f', error_big_nist));
end

if error_small_nist < 0.25
    disp(sprintf('Success! The SMALL classifier performs with a NIST error of %f', error_small_nist));
else
    disp(sprintf('The SMALL classifier does not fulfill the requirements: NIST error of %f', error_small_nist));
end


%% Testing with custom digits

digits = handwritten_data('Digit_scanner/good_digits');

labels_big = digits*classifier_big*labeld;
labels_small = digits*classifier_small*labeld;

show_digits(digits, labels_big)
show_digits(digits, labels_small)