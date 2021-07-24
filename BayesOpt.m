%Parameters
Inputs = Predictors;
Outputs = Response;

%Setting up hyperparameters that need to be optimized 

minHiddenLayerSize = 1;
maxHiddenLayerSize = 20;
hiddenLayerSizeRange = [minHiddenLayerSize maxHiddenLayerSize];
optimVars = [optimizableVariable('NumHL', [1 3],'Type','integer')
             optimizableVariable('Layer1Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('Layer2Size',hiddenLayerSizeRange,'Type','integer')
             optimizableVariable('TrainFcn0', {'trainlm' 'trainbr' 'trainbfg' 'traincgb' 'traincgf' 'traingd' 'traingda' 'traingdm' 'traingdx' 'trainoss' 'trainrp' 'trainscg' 'trainb' 'trains'},'Type','categorical')               
             optimizableVariable('nEpochs',[50 2000],'Type','integer')
             optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
             optimizableVariable('Momentum',[0.8 0.98])
             optimizableVariable('TrainFcn1',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')
             optimizableVariable('TrainFcn2',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')]
             optimizableVariable('TrainFcn3',{'logsig' 'tansig' 'satlins' ...
             'purelin' 'poslin' 'satlin' 'compet' 'elliotsig' 'hardlim'...
             'hardlims' 'netinv' 'radbas' 'radbasn' 'softmax' 'tribas'},'Type','categorical')
             optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
        
%Bayesion Optimization

ObjFcn = makeObjFcn(Inputs, Targets);

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',300,...
    'MaxTime',1*60*60,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);

function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
    ObjFcn = @valPerfFind;
      function valPerf = valPerfFind(optVars)
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
                     'ValidationFrequency',validationFrequency), ...
                     
                    
          
          %Training Function 
          trainFcn = char(optVars.TrainFcn0);  % Bayesian Regularization backpropagation.
          % Create a Fitting Network
          layer1_size = optVars.Layer1Size;
          layer2_size = optVars.Layer2Size;
          maxEpochs = optVars.nEpochs;
          LR = optVars.InitialLearnRate;
          Momentum = optVars.Momentum;
          TrainFcn1 = char(optVars.TrainFcn1);
          TrainFcn2 = char(optVars.TrainFcn2);
          layers = [featureInputLayer
                    layer1
                    layer2
                    layer3
                    softmaxLayer
                    classificationLayer]
              
         
          if optVars.NumHL == 2 
            hiddenLayerSizes = [layer1_size layer2_size];
          else 
              hiddenLayerSizes = [layer1_size];
          end
          % Specifying activation function at each layer
          net = trainNetwork(hiddenLayerSizes,trainFcn);

          % Setup Division of Data for Training, Validation, Testing
          net.divideParam.trainRatio = 70/100;
          net.divideParam.valRatio = 15/100;
          net.divideParam.testRatio = 15/100;
          % Activation function for hidden layers
          if optVars.NumHL == 2 
              net.layers{1}.transferFcn = TrainFcn1;  % Hidden layer 1
              net.layers{2}.transferFcn = TrainFcn2;  % Hidden layer 2
              net.layers{3}.transferFcn = 'purelin';  % Output layer
          else 
              net.layers{1}.transferFcn = TrainFcn1;  % Hidden layer 
              net.layers{2}.transferFcn = 'purelin';
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
          valPerf = perform(net,YTrain,YPredicted);
         
      end
  end