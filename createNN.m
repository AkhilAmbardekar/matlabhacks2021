clear
clc

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

% Predictors = [duration; temperature; humidity; visibility; windSpeed; ...
%     crossing; mStop; trafficSignal; weatherCondOrd; dayNightOrd]';

Predictors = [duration]';
    
Response = categorical(severity);

% netData = table(Predictors, Response);

[numData, numFeatures] = size(Predictors);

% trainSplit = 0.70 ;
% idx = randperm(numData)  ;
% XTrain = A(idx(1:round(trainSplit*numData)),:) ; 
% XTest = A(idx(round(trainSplit*numData)+1:end),:) ;

[trainInd,valInd,testInd] = dividerand(numData,0.7,0.15,0.15);

XTrain = Predictors(trainInd, :);
YTrain = Response(trainInd, :);

XValid = Predictors(valInd, :);
YValid = Response(valInd, :);

XTest = Predictors(testInd, :);
YTest = Response(testInd, :);

layers = [featureInputLayer(numFeatures, "Normalization", "none")... % input layer
    fullyConnectedLayer(6)... % dense hidden layer
        reluLayer... % 
    fullyConnectedLayer(4)... % classification with 4 classes
        softmaxLayer... % must be fixed to softmax
            classificationLayer]; % output layer
        
options = trainingOptions("adam", ...
    "MaxEpochs",1,...
    "InitialLearnRate",1e-2,...
    "Verbose",false,...
    "ValidationData",{XValid,YValid},...
    "Plots","training-progress");
        
net = trainNetwork(XTrain, YTrain, layers, options);

YPredict = classify(net, XTest);

testError = perform(net, double(YTest), double(YPredict));

