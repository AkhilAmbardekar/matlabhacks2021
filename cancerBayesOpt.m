% for reproducibility
clear
clc
close all

rng('default');
s = rng;

% read all data
cancerData = readtable("cancer_reg.csv");

% output
Targets = cancerData.TARGET_deathRate';

% inputs
Inputs = cancerData;

% remove columns with missing data and target
Inputs = removevars(Inputs,{'TARGET_deathRate', 'MedianAgeMale', 'MedianAgeFemale', ...
                            'binnedInc', 'avgAnnCount',  'avgDeathsPerYear', ...
                            'Geography', 'PctSomeCol18_24', 'PctBachDeg25_Over', ...
                            'PctEmployed16_Over', 'PctPrivateCoverageAlone'});
Inputs = table2array(Inputs)';

% set range for hidden layer size
minHiddenLayerSize = 1;
maxHiddenLayerSize = 50;
hiddenLayerSizeRange = [minHiddenLayerSize maxHiddenLayerSize];

% set variables to optimize
optimVars = [optimizableVariable('numHL',[1, 3],'Type','integer')
             optimizableVariable('Layer1Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('Layer2Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('Layer3Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('ActivFun1',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')
             optimizableVariable('ActivFun2',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')
             optimizableVariable('ActivFun3',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')
             ];

 % optimize the network's layer sizes and activation functions
ObjFun = makeObjFun(Inputs, Targets);
BayesObj = bayesopt(ObjFun,optimVars,... % initializing Bayes optimizer
    'MaxObj',300,...
    'MaxTime',60*60,...
    'IsObjectiveDeterministic',true,...
    'UseParallel',true);

% find the best network and load it and get its test error
bestNetI = BayesObj.IndexOfMinimumTrace(end);
fileName = BayesObj.UserDataTrace{bestNetI};
load(fileName);
Predict = net(Inputs);
testErr = perform(net,Targets,Predict);
testErr

%% 
% The matrix format is as follows:
% 
% X  - RxQ matrix
% 
% Y  - UxQ matrix.
% 
% Where:
% 
% Q  = number of samples
% 
% R  = number of elements in the network's input
% 
% U  = number of elements in the network's output

function ObjFun = makeObjFun(Inputs,Targets)
    ObjFun = @valErrorFun;
      function [error,cons,fileName] = valErrorFun(optVars)
          % bayesian regularization to minimize overfitting
          trainFcn = 'trainbr';
          
          % layer sizes
          layer1size = optVars.Layer1Size;
          layer2size = optVars.Layer2Size;
          layer3size = optVars.Layer3Size;
            
          % constant training params
          epochsNum = 50;
          lr = 0.01;
          momentum = 0.9;
          
          % activation functions from categorical to char
          ActivFun1 = char(optVars.ActivFun1);
          ActivFun2 = char(optVars.ActivFun2);
          ActivFun3 = char(optVars.ActivFun3);
          
          % different network structure depending on # of hidden layers
          if optVars.numHL == 1
              hiddenLayerSizes = [layer1size]; 

              net = fitnet(hiddenLayerSizes,trainFcn);

              net.layers{1}.transferFcn = ActivFun1;
              
          elseif optVars.numHL == 2
              hiddenLayerSizes = [layer1size, layer2size]; 

              net = fitnet(hiddenLayerSizes,trainFcn);

              net.layers{1}.transferFcn = ActivFun1;
              net.layers{2}.transferFcn = ActivFun2;             
              
          else
              hiddenLayerSizes = [layer1size, layer2size, layer3size]; 

              net = fitnet(hiddenLayerSizes,trainFcn);

              net.layers{1}.transferFcn = ActivFun1;
              net.layers{2}.transferFcn = ActivFun2;
              net.layers{3}.transferFcn = ActivFun3;              
              
          end
          
          % train test valid split
          net.divideParam.trainRatio = 70/100;
          net.divideParam.valRatio = 15/100;
          net.divideParam.testRatio = 15/100;          

          % measure using mean square error
          net.performFcn = 'mse';
          net.performParam.normalization = 'standard';

          % set training params
          net.trainParam.epochs = epochsNum;
          net.trainParam.lr = lr;
          net.trainParam.mc = momentum; 
          
          % train
          [net,~] = train(net,Inputs,Targets);
            
          % evaluate using error
          Predict = net(Inputs);
          error = perform(net,Targets,Predict); % defines the parameters and values of the current performance function
          fileName = num2str(error) + ".mat";
          save(fileName,'net','error')
          cons = [];
      end
end