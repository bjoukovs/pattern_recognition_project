clear all, close all, prwaitbar off

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
dataset_small = load('datasets/small_dataset.mat');
train_small = dataset_small.train;
tst_small = dataset_small.tst;

nist_testing_small = load('datasets/small_nist_eval.mat');
nist_data_small = nist_testing_small.DATA2;


%% Training the big classifier

classifier_big = gen_classifier_big(train_big, tst_big);



%% Training the small classifier

classifier_small = gen_classifier_small(train_small, tst_small);


%% Testing the classifiers with NIST_EVAL

error_big = nist_eval(nist_data_big, classifier_big);
error_small = nist_eval(nist_data_small, classifier_small);

if error_big < 0.05
    disp(sprintf('Success! The BIG classifier performs with a NIST error of %f', error_big));
else
    disp(sprintf('The BIG classifier does not fulfill the requirements: NIST error of %f', error_big));
end

if error_small < 0.2
    disp(sprintf('Success! The SMALL classifier performs with a NIST error of %f', error_small));
else
    disp(sprintf('The SMALL classifier does not fulfill the requirements: NIST error of %f', error_small));
end


%% Testing with custom digits

digits = handwritten_data('Digit_scanner/good_digits');

labels_big = labeld(digits*classifier_big);
labels_small = labeld(digits*classifier_small);

show_digits(digits, labels_big)
show_digits(digits, labels_small)