%Parameters
clear
clc
close all

% read all data
allData = readtable("crashes_small.xlsx");

% extract output
severity = allData.Severity;

% extract inputs
startTime = allData.Start_Time;
endTime = allData.End_Time;
duration = endTime - startTime;
duration = duration';
duration = datenum(duration);
% duration = normalize(duration);

temperature = allData.Temperature_F_';
% temperature = normalize(temperature);
humidity = allData.Humidity___';
% humidity = normalize(humidity);
visibility = allData.Visibility_mi_';
% visibility = normalize(visibility);
windSpeed = allData.Wind_Speed_mph_';
% windSpeed = normalize(windSpeed);
crossing = allData.Crossing';
crossing = double(crossing);
mStop = allData.Stop';
mStop = double(mStop);
trafficSignal = allData.Traffic_Signal';
trafficSignal = double(trafficSignal);

% ordinal encoding for weather
weatherCond = allData.Weather_Condition';
weatherCond = categorical(weatherCond);
weatherCondOrd = grp2idx(weatherCond)';
% weatherCondOrd = normalize(weatherCondOrd);% use unique to find how it has been transcribed

dayNight = allData.Sunrise_Sunset';
dayNight = categorical(dayNight);
dayNightOrd = grp2idx(dayNight)';
dayNightOrd = dayNightOrd - ones(1, 50000);
% dayNightOrd = normalize(dayNightOrd);
 
Predictors = [duration; crossing; mStop; trafficSignal; weatherCondOrd; dayNightOrd]';

Response = categorical(severity);

[numData, numFeatures] = size(Predictors);

[trainInd,valInd,testInd] = dividerand(numData,0.7,0.15,0.15);

XTrain = Predictors(trainInd, :);
YTrain = Response(trainInd, :);

XValid = Predictors(valInd, :);
YValid = Response(valInd, :);

XTest = Predictors(testInd, :);
YTest = Response(testInd, :);

%Setting up hyperparameters that need to be optimized 

minHiddenLayerSize = 4;
maxHiddenLayerSize = 30;
hiddenLayerSizeRange = [minHiddenLayerSize maxHiddenLayerSize];
optimVars = [optimizableVariable('NumHL', [1 3],'Type','integer') 
             optimizableVariable('Layer1Size',hiddenLayerSizeRange,'Type','integer') 
             optimizableVariable('Layer2Size',hiddenLayerSizeRange,'Type','integer') 
             optimizableVariable('Layer3Size', hiddenLayerSizeRange, 'Type','integer') 
             optimizableVariable('layer1',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical' )
             optimizableVariable('layer2',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical')
             optimizableVariable('layer3',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical')];
        
%Bayesion Optimization

ObjFcn = makeObjFcn(XTrain, YTrain, XValid, YValid);

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',300,...
    'MaxTime',1*60*60,...
    'IsObjectiveDeterministic',true,...
    'UseParallel',false);


%Evaluate Final Network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError


function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
    ObjFcn = @valErrorFun;
      function testError = valErrorFun(optVars)
          options = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.01, ...
                     'Momentum', 0.9, ...
                     'MaxEpochs', 50, ... 
                     'L2Regularization',0.01, ...
                     'Plots','training-progress', ... 
                     'Shuffle','every-epoch', ...
                     'ValidationData',{XValidation,YValidation});
                     
          if optVars.NumHL == 3
              layers = [featureInputLayer(numFeatures, "Normalization", "none")... % input layer
                        fullyConnectedLayer(optVars.Layer1Size)... % dense hidden layer
                        optVars.layer1... % 
                        fullyConnectedLayer(optVars.Layer2Size)
                        optVars.layer2
                        fullyConnectedLayer(optVars.Layer3Size)
                        optVars.layer3
                        fullyConnectedLayer(4)... % classification with 4 classes
                        softmaxLayer... % must be fixed to softmax
                        classificationLayer]; % output layer
          elseif optVars.NumHL == 2
              layers = [featureInputLayer(numFeatures, "Normalization", "none")... % input layer
                        fullyConnectedLayer(optVars.Layer1Size)... % dense hidden layer
                        optVars.layer1... % 
                        fullyConnectedLayer(optVars.Layer2Size)
                        optVars.layer2
                        fullyConnectedLayer(4)... % classification with 4 classes
                        softmaxLayer... % must be fixed to softmax
                        classificationLayer]; % output layer
          else
              layers = [featureInputLayer(numFeatures, "Normalization", "none")... % input layer
                        fullyConnectedLayer(optVars.Layer1Size)... % dense hidden layer
                        optVars.layer1... % 
                        fullyConnectedLayer(4)... % classification with 4 classes
                        softmaxLayer... % must be fixed to softmax
                        classificationLayer]; % output layer
          end
         
            net = trainNetwork(XTrain, YTrain, layers, options);

            YPredict = classify(net, XTest);

            testError = perform(net, double(YTest), double(YPredict));

      end
  end