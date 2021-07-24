%Parameters
Inputs = Predictors;
Outputs = Response;
ValX;
ValY;

%Setting up hyperparameters that need to be optimized 

minHiddenLayerSize = 1;
maxHiddenLayerSize = 20;
hiddenLayerSizeRange = [minHiddenLayerSize maxHiddenLayerSize];
optimVars = [optimizableVariable('NumHL', [1 3],'Type','integer')
             optimizableVariable('Layer1Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('Layer2Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('Layer3Size', hiddenLayerSizeRange, 'Type','integer')
             optimizableVariable('nEpochs',[50 2000],'Type','integer')
             optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
             optimizableVariable('Momentum',[0.8 0.98])
             optimizableVariable('layer1',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical' )
             optimizableVariable('layer2',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical')
             optimizableVariable('layer3',{'reluLayer' 'leakyReluLayer' ...
             'clippedReluLayer','eluLayer', 'tanhLayer','swishLayer'},'Type','categorical')
             optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
        
%Bayesion Optimization

ObjFcn = makeObjFcn(Inputs, Targets, ValX, ValY);

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',300,...
    'MaxTime',1*60*60,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);


%Evaluate Final Network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError


function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
    ObjFcn = @valErrorFun;
      function valError = valErrorFun(optVars)
          miniBatchSize = 128;
          validationFrequency = floor(numel(YTrain)/miniBatchSize)
          options = trainingOptions('sgdm', ...
                    'InitialLearnRate', optVars.InitialLearnRate, ...
                     'Momentum', optVars.Momentum, ...
                     'MaxEpochs', optVars.nEpochs, ... 
                     'L2Regularization',optVars.L2Regularization, ...
                     'LearnRateSchedule','piecewise', ...
                     'Plots','training-progress', ... 
                      'Shuffle','every-epoch', ...
                     'ValidationData',{XValidation,YValidation}, ...
                     'ValidationFrequency',validationFrequency));
                     
                    
          
          
          if optVars.NumHL == 3
              layers = [featureInputLayer
                        fullyConnectedLayer(optVars.Layer1Size)
                        optVars.layer1
                        fullyConnectedLayer(optVars.Layer2Size)
                        optVars.layer2
                        fullyConnectedLayer(optVars.Layer3Size)
                        optVars.layer3
                        softmaxLayer
                        classificationLayer]
          else if optVars.Num HL == 2
              layers = [featureInputLayer
                        fullyConnectedLayer(optVars.Layer1Size)
                        optVars.layer1
                        fullyConnectedLayer(optVars.Layer2Size)
                        optVars.layer2
                        softmaxLayer
                        classificationLayer]
          else
              layers = [featureInputLayer
                    fullyConnectedLayer(optVars.Layer1Size)
                    optVars.layer1
                    softmaxLayer
                    classificationLayer]
              end
         
           YPredicted = classify(trainedNet,XValidation);
           valError = 1 - mean(YPredicted == YValidation);
        
         
      end
  end