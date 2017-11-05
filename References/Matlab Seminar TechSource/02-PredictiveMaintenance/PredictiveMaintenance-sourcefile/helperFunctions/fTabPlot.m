function fTabPlot(Cycles,Table,RunAnim,AxesTab)

if nargin<3
    RunAnim = false;
end
if nargin<4
    AxesTab = [2,3];
end

%% Change Default Figure Settings
%  make the default figure position large so that the graphs are more
%  legible.
spos = get(0,'screensize');
set(0,'defaultFigurePosition',[spos(3)*.2 spos(4)*.2 spos(3)*.6 spos(4)*.6]);

%% Initialize canvas

NAxesTab = prod(AxesTab);
NCycles = size(Table,1);
NSignals = size(Table,2);
NTabs = length(1:NAxesTab:size(Table,2));

f=figure(1);
clf;
tabgp = uitabgroup(f);
tabHdl = gobjects(1,NTabs);    % tab handle
axHdl = gobjects(1,NSignals);  % axes handle
lineHdl = gobjects(1,NSignals);   % line handle

%% Define color map

colorMap = [...
    0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
NColors = size(colorMap,1);

%%
indTab = 0;
for indSig =1:NAxesTab:size(Table,2) % ks: signal index
    
    % Add title to each tab
    indSigTab = indSig:min([NSignals,indSig+NAxesTab-1]);
    if length(indSigTab)>1
        tabtt = sprintf('Signals %d-%d',indSig,indSigTab(end));
    else
        tabtt = sprintf('Signal %d',indSig);
    end
    
    indTab = indTab + 1;
    tabHdl(indTab) = uitab(tabgp,'Title',tabtt);
    
    % Add NAxesTab axes to each tab
    for indAx = 1:min([length(indSigTab),NAxesTab])
        axHdl(indSigTab(indAx)) = subplot(AxesTab(1),AxesTab(2),indAx,'Parent',tabHdl(indTab));
        ax = gca;
        ax.Color = 'none';
        
        lineColor = colorMap(rem(indAx-1,NColors)+1,:);
        if RunAnim
            lineHdl(indSigTab(indAx)) = line(Cycles,nan(NCycles,1),'Color',lineColor,'LineWidth',2);
        else
            lineHdl(indSigTab(indAx)) = line(Cycles,Table{:,indSigTab(indAx)},'Color',lineColor,'LineWidth',2);
        end
        %         set(axHdl(kS(ka)),'XLim',XLim,'YLim',YLim(kS(ka),:))
        title(Table.Properties.VariableNames{indSigTab(indAx)},'Color',lineColor)
        xlabel('Cycles')
    end
end

if RunAnim
    for kCycle = 1:NCycles
        for ksignal = 1:NSignals
            lineHdl(ksignal).YData(kCycle) = Table{kCycle,ksignal};
        end
        drawnow limitrate
    end
end
shg
