clear
clc
close all

cancerData = readtable("cancer_reg.csv");

% output
targets = cancerData.PctPrivateCoverageAlone';

% inputs
inputs = cancerData;
inputs = removevars(inputs,{'TARGET_deathRate', 'MedianAgeMale', 'MedianAgeFemale', ...
                            'binnedInc', 'avgAnnCount',  'avgDeathsPerYear', ...
                            'Geography', 'PctSomeCol18_24', 'PctBachDeg25_Over', ...
                            'PctEmployed16_Over', 'PctPrivateCoverageAlone'});
inputs = table2array(inputs)';