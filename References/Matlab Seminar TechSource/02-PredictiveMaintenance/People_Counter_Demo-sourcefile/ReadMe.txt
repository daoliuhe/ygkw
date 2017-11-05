Creating a Cloud Based People Counter Using MATLAB 
------------------------------------------------
Description: 
This projects utilizes the Computer Vision System Toolbox, ThingSpeak and a USB webcam 
to count the number of people that pass by the webcam. 

To Run This Demo:
Step 1: Plug the USB Webcam into the computer running MATLAB
Step 2: Download the MultiObjectTrackerKLT.m file from File Exchange (download from link below)
https://www.mathworks.com/matlabcentral/fileexchange/47105-detect-and-track-multiple-faces 
Step 3: Create your own ThingSpeak channel
Step 4: Change the channel ID, the read key and the write key in the code 
(Line 16 of PeopleCounterDemo.m and Lines 3-4 of logThingSpeakData.m)
Step 5: Run PeopleCounterDemo.m 

Notes:
1) To have better command of the results, a new ThingSpeak channel should
be created to read and write to. The channel used in the script is a public 
channel so the count data incorporates the use of everyone who has used
the provided MATLAB script. 

2) The algorithm may sometimes detect inanimate objects as faces if they
resemble the facial structure of a person.  