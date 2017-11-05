function out = createThresholds(in,catThreshold,orderedCategory)

numCategory = 1:length(catThreshold)+1;
in.TTF = ones(height(in),1);
in.TTF(in.Time<=-catThreshold(1)) = 2;
in.TTF(in.Time<=-catThreshold(2)) = 3;
in.TTF(in.Time<=-catThreshold(3)) = 4;
out = categorical(in.TTF,numCategory,orderedCategory,'Ordinal',true);
