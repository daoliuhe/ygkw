function fTabRealTime(Cycles,Table,CatNames,Threshold,Label,LINE,RUN,AxesTab)

if nargin<5
    SCATTER = false;
else
    if ~isequal(CatNames',Label)
        Label = reordercats(Label,CatNames);
    end
    SCATTER = true;
end
if nargin<6
    LINE = true;
end
if nargin<7
    RUN = false;
end
if nargin<8
    AxesTab = [2,3];
end

%% Change Default Figure Settings
%  make the default figure position large so that the graphs are more
%  legible.
spos = get(0,'screensize');
f = figure;
f.Position = [spos(3)*.08 spos(4)*.15 spos(3)*.8 spos(4)*.7];
set(f,'ToolBar','none','MenuBar','none')

%%

NAxesTab = prod(AxesTab);
NCycles = size(Table,1);
NSignals = size(Table,2);
NTabs = length(1:NAxesTab:NSignals);
Nc = length(CatNames); % number of categories

map = [192 0 0; ... % urgent: red
    255 255 0; ...  % short:  yellow
    146 208 80; ... % medium:
    18 86 135]/255; % long:   blue

catNrs = double(Label);

f.Colormap = map;

tabgp = uitabgroup(f);
tabHdl = gobjects(1,NTabs);    % tab handle
lightHdl = gobjects(1,NTabs); % one light per tab
titlHdl = gobjects(1,NTabs);     % Title handles
titlHdl4Light = gobjects(1,NTabs);     % Title handles
cb = gobjects(1,NTabs);     % colorbar handles
axHdl = gobjects(1,NSignals);  % axes handle

if SCATTER
    scatHdl = gobjects(1,NSignals);   % scatter plot handle
end
if LINE
    lineHdl = gobjects(1,NSignals);   % line handle
    LinePropt = {'Color',.7*[1 1 1],'LineWidth',1};
end

ktab = 0;
for ks =1:NAxesTab:NSignals % ks: signal index
    ktab = ktab + 1;
    kS = ks:min([NSignals,ks+NAxesTab-1]);
    if length(kS)>1
        tabtt = sprintf('Signals %d-%d',ks,kS(end));
    else
        tabtt = sprintf('Signal %d',ks);
    end
    tabHdl(ktab) = uitab(tabgp,'Title',tabtt);
    tabHdl(ktab).BackgroundColor = 'k';
    
    %%
    titlHdl(ktab) = subplot(AxesTab(1)+1,AxesTab(2),1,'Parent',tabHdl(ktab),'Visible', 'off');
    posVal = titlHdl(ktab).Position;
    posVal(4) = posVal(4)*0.15;
    titlHdl(ktab).Position = posVal;
    set(get(titlHdl(ktab),'Title'),'Visible','on');
    title(titlHdl(ktab),sprintf('Realtime Prediction\nof Time to Failure\nbased on %d Sensor Readings', NSignals),...
        'FontSize',18,'Color','w');
    
    lightHdl(ktab) = subplot(AxesTab(1)+1,AxesTab(2),2,'Parent',tabHdl(ktab));
    fTrafLight('w');
    
    titlHdl4Light(ktab) = subplot(AxesTab(1)+1,AxesTab(2),3,'Parent',tabHdl(ktab),'Visible', 'off');
    posVal = titlHdl4Light(ktab).Position;
    posVal(1) = posVal(1)*0.9;
    posVal(4) = posVal(4)*0.25;
    titlHdl4Light(ktab).Position = posVal;
    set(get(titlHdl4Light(ktab),'Title'),'Visible','on');
    title(titlHdl4Light(ktab),'No Signal','FontSize',20,'Color','w');
    
    %%
    
    
    for ka = 1:min([length(kS),NAxesTab])
        axHdl(kS(ka)) = subplot(AxesTab(1)+1,AxesTab(2),ka+AxesTab(2),'Parent',tabHdl(ktab));
        axHdl(kS(ka)).Color = 'none';
        axHdl(kS(ka)).XColor = 'w';
        axHdl(kS(ka)).YColor = 'w';
        hold on
        if RUN
            if LINE
                lineHdl(kS(ka)) = line(Cycles,nan(NCycles,1),LinePropt{:});
            end
            if SCATTER
                scatHdl(kS(ka)) = scatter(Cycles,nan(NCycles,1),[],Label,'filled');
            end
        else
            if LINE
                lineHdl(kS(ka)) = line(Cycles,Table{:,kS(ka)},LinePropt{:});
            end
            if SCATTER
                scatHdl(kS(ka)) = scatter(Cycles,Table{:,kS(ka)},[],Label,'filled');
            end
        end
        hold off
        title(Table.Properties.VariableNames{kS(ka)},'Color','w')
        xlabel('Cycles')
        set(axHdl(kS(ka)),'CLim',[1 Nc]);
    end
    cb(ktab)=axes('Units','normalized','Pos',[.925 .125 .025 .75], ...
        'Parent',tabHdl(ktab));
    imagesc((0:Nc-1)','Parent',cb(ktab))
    cb(ktab).XTick = [];
    cb(ktab).YTick = 1:Nc;
    cb(ktab).YTickLabel = CatNames;
    cb(ktab).YColor = 'w';
    cb(ktab).YTickLabelRotation = 90;
    cb(ktab).YAxisLocation = 'right';
    
end
shg
if RUN
    for kCycle = 1:NCycles
        for ksignal = 1:NSignals
            if LINE
                lineHdl(ksignal).YData(kCycle) = Table{kCycle,ksignal};
            end
            if SCATTER
                scatHdl(ksignal).YData(kCycle) = Table{kCycle,ksignal};
            end
        end
        mycatNr = catNrs(kCycle);
        myfillColor = map(mycatNr,:);
        for indtab = 1:ktab            
            lightHdl(indtab).Children.FaceColor = myfillColor;
            lightHdl(indtab).Children.EdgeColor = myfillColor;

            if catNrs(kCycle) > 2
                title(titlHdl4Light(indtab),{'Normal'},'FontSize',20,'Color',myfillColor);
            elseif catNrs(kCycle) == 2
                title(titlHdl4Light(indtab),...
                    {sprintf('Warning,\nplan maintenance\nwithin %d operation cycles',...
                    Threshold(2))},'FontSize',18,'Color',myfillColor);          
            elseif kCycle >= 6 && (all(catNrs(kCycle:-1:kCycle-2) == 1))
                title(titlHdl4Light(indtab),...
                    {sprintf('Alarm\nmaintenance needed, alarm sent to\n Yi.Wang@mathworks.com')},...
                    'FontSize',18,'Color',myfillColor, ...
                    'Interpreter','none');
                if (all(catNrs(kCycle-3:-1:kCycle-5) > 1)) && indtab ==1
                                        writetable(Table(1:kCycle,:),'SensorData.csv','Delimiter',',');
                                        fSendOutlookMail('Yi.Wang@mathworks.com',...
                                            sprintf('Alarm: Equipment is predicticted to fail within %d operation cycles', Threshold(1)),...
                                            ['Date:' datestr(datetime('now')) sprintf('.\n Please plan maintenance asap')],...
                                            {fullfile(pwd,'SensorData.csv')});
                end
            else
                title(titlHdl4Light(indtab),{'Alarm'},'FontSize',20,'Color',myfillColor);
            end
            if kCycle == NCycles
                lightHdl(indtab).Children.FaceColor = 'w';
                lightHdl(indtab).Children.EdgeColor = 'w';
                title(titlHdl4Light(indtab),{'No Signal'},'FontSize',20,'Color','w');
            end
        end
%         drawnow limitrate
        pause(0.15)
    end
end
