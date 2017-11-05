% logThingSpeakData
% This function is used by PeopleCounterDemo.m to write data to a
% ThingSpeak channel. 
%
% Copyright 2016 The MathWorks, Inc

function logThingSpeakData(tracker)

ChannelID = 181351;
WriteKeyVal = 'AZO3MR3XOTH9GQ0M';


if ischar(tracker) && strcmp(tracker,'reset')
    thingSpeakWrite(ChannelID,[0 0],'WriteKey',WriteKeyVal)
else
    data = [0 0];
    data(1) = tracker.NextId-1;
    data(2) = numel(tracker.BoxIds);
    
    try
    thingSpeakWrite(ChannelID,data,'WriteKey',WriteKeyVal,'Timeout',5)
    catch
    end
end