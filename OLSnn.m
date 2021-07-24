clear
clc
close all


%read all data
allData = readtable("cancer_reg.csv");

%extract output
mortRate = allData.TARGET_deathRate;

%extract inputs
