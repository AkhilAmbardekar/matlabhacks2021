% for reproducibility
clear
clc
close all


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


%% Configuration of Particle Swarm Algorithm
%[number of hidden layers, layer 1 size, layer 2 size, layer 3 size, activation function 1, activation function 2, activation function 3]

lb = [1 1  1  1  1  1  1]; % lower bound
ub = [3 50 50 50 15 15 15];  % upper bound
  
options = optimoptions(@particleswarm, 'MaxTime', 1,'Display','iter', 'PlotFcn', 'pswplotbestf','MaxIterations',200, 'SwarmSize', 20);
                   

%% Running Particle Swarm Algorithm
tic

%Objective Function
ObjFcn = makeObjFcn(Inputs, Targets);

[X, fbest, exitflag,out] = particleswarm(ObjFcn,7,lb,ub,options);
toc
disp(X)
disp(fbest)
actFunc = {'logsig', 'tansig', 'satlins','purelin', 'poslin', 'satlin', 'compet', 'elliotsig', 'hardlim', 'hardlims', 'netinv', 'radbas', 'radbasn', 'softmax', 'tribas'}; %ub 15





%% Output Function to Optimize
function ObjFcn = makeObjFcn(XTrain,YTrain)
    ObjFcn = @valErrorFun;
    function valError = valErrorFun(OptVars)       
          %Choose 'trainbr' training function to minimize overfitting          
          trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.
          
          % Set layer sizes
          layer1_size = round(OptVars(2));
          layer2_size = round(OptVars(3));
          layer3_size = round(OptVars(4));
          
          %Set number of epochs, learning rate, and momentum
          maxEpochs = 50;
          LR = 0.01;
          Momentum = 0.9;
          
          %Set activation functions for each layer 
          actFunc = {'logsig', 'tansig', 'satlins','purelin', 'poslin', 'satlin', 'compet', 'elliotsig', 'hardlim', 'hardlims', 'netinv', 'radbas', 'radbasn', 'softmax', 'tribas'}; %ub 15
          actF1 = round(OptVars(5));
          actF2 = round(OptVars(6));
          actF3 = round(OptVars(7));
          TrainFcn1 = actFunc(actF1);
          TrainFcn2 = actFunc(actF2);
          TrainFcn3 = actFunc(actF3);
          
          %Determine architecture of network based on number of hidden
          %layers
          if round(OptVars(1)) == 3 
            hiddenLayerSizes = [layer1_size layer2_size layer3_size];
          elseif round(OptVars(1)) == 2
            hiddenLayerSizes = [layer1_size layer2_size];
          else
             hiddenLayerSizes = [layer1_size];
          end
          
          % Create neural network 
          net = fitnet(hiddenLayerSizes,trainFcn);
    
          % Setup Division of Data for Training, Validation, Testing
          net.divideParam.trainRatio = 75/100;
          net.divideParam.valRatio = 0/100;
          net.divideParam.testRatio = 25/100;
          
          % Activation function for hidden layers
          if round(OptVars(1)) == 3  
              net.layers{1}.transferFcn = char(TrainFcn1);  
              net.layers{2}.transferFcn = char(TrainFcn2);  
              net.layers{3}.transferFcn = char(TrainFcn3);  
          elseif round(OptVars(1)) == 2
              net.layers{1}.transferFcn = char(TrainFcn1);   
              net.layers{2}.transferFcn = char(TrainFcn2);
          else
             net.layers{1}.transferFcn = char(TrainFcn1);   
          end
          
          % performance function
          net.performFcn = 'mse';
          net.performParam.normalization = 'standard';
          
          % Train the Network
          net.trainParam.showWindow = true;
          net.trainParam.showCommandLine = false;
          net.trainParam.lr = LR;
          net.trainParam.mc = Momentum;
          net.trainParam.epochs = maxEpochs;
          [net,~] = train(net,XTrain,YTrain);
          
          % Test the Network
          YPredicted = net(XTrain);
          valError = perform(net,YTrain,YPredicted);
        end
  end
