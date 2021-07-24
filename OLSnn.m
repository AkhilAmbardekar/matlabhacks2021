clear
clc
close all


%read all data
allData = readtable("cancer_reg.csv");

%extract output
mortRate = allData.TARGET_deathRate;

%extract inputs
medIncome = allData.medIncome;
popEst2015 = allData.popEst2015;
povertyPercent = allData.povertyPercent;
studyPerCap = allData.studyPerCap;
medAge = allData.MedianAge;
avgHouse = allData.AvgHouseholdSize;
