function MonitoringDashboard(engNum)
if isdeployed()
    engNum = str2double(engNum);
end
%#function ClassificationKNN

if ~any(engNum==1:100)
    error('Input must be integer value from 1 to 100')
end
%addpath(fullfile(pwd,'helperFunctions'))
%% load trained model
load('trainedmodel.mat')

%% read-in one file
%filename = 'train_FD001_Unit_';
%file = fullfile(pwd,'Data',[filename num2str(engNum) '.csv']);
file = ['Data/train_FD001_Unit_' num2str(engNum) '.csv'];
Data = readtable(file,'ReadVariableNames',true);

%% Select relevant variable names based on visualization
VariableNames = {'Unit' 'Time' 'LPCOutletTemp' 'HPCOutletTemp' ...
    'LPTOutletTemp' 'TotalHPCOutletPres' 'PhysFanSpeed' ...
    'PhysCoreSpeed' 'StaticHPCOutletPres' 'FuelFlowRatio'...
    'CorrFanSpeed' 'CorrCoreSpeed' 'BypassRatio'...
    'BleedEnthalpy' 'HPTCoolantBleed' 'LPTCoolantBleed'};
SensorNames = VariableNames(3:end);
filterWindow = 5;
Threshold = [50, 125, 200];                     % thresholds
CatNames = {'urgent','short','medium','long'};  % categories

%% Prepare data needed for the prediction
testData = fPreprocess(Data,SensorNames,filterWindow);
[~,testLabel] = fLabel(testData.Time,Threshold,CatNames);
D = testData(:,SensorNames); 

%% Predict the labels using the exported MATLAB function

%load trainedmodel
predictedLabel = predict(trainedClassifier,D);    
fTabRealTime(testData.Time,D,CatNames,Threshold,predictedLabel,1,1)

wrongPredictions = find(predictedLabel~=testLabel);
fprintf('Engine %3d: ',testData.Unit(1))
if isempty(wrongPredictions)
    fprintf('100%% correct predictions\n')
else
    fprintf(2,'%d prediction error(s)\n',length(wrongPredictions));
end
