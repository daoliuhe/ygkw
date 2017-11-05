function [Time,Label] = fLabel(Time, Threshold, CatNames)

%% Shift timeseries so failure occurs at t=0
% The time series begin an unknown amount of time after initial use and run
% until failure conditions. As such they have variable length, but always
% end in failure. Shift the signals so that time t=0 is when failure occurs
% and negative times represent cycles prior to failure.

Time = Time - max(Time);

%% Label sensor data in categories based on number of cycles until failure
% 1 - urgent: very small number of cycles until failure
% 2 - short:  small      number of cycles until failure
% 3 - medium: medium     number of cycles until failure
% 4 - long:   large      number of cycles until failure

% Create a new table with a new variable
Label = zeros(length(Time),1);

% if fail within 50 cycTles then classify as urgent
Label((0 >= Time) & (Time > -Threshold(1))) = 1;
% if fail between 50 and 125 cycles then classify as short
Label((-Threshold(1) >= Time) & (Time > -Threshold(2))) = 2;
% if fail between 125 and 200 cycles then classify as medium
Label((-Threshold(2) >= Time) & (Time > -Threshold(3))) = 3;
% if fail greater than 200 cycles then classify as long
Label(-Threshold(3) >= Time ) = 4;

%% Convert numerical to categorical 

Ncat = length(CatNames);
Label = categorical(Label,(1:Ncat),CatNames);