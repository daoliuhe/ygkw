function [data] = fPreprocess(data,sensorNames,filterWindow)

%% Filter sensor data with moving average to get rid of sensor noise
a = 1;
b = (1/filterWindow)*ones(1,filterWindow);
data{:,sensorNames} = filter(b,a,data{:,sensorNames});

%% Exclude data out of filter window
data = data(filterWindow:end, :);