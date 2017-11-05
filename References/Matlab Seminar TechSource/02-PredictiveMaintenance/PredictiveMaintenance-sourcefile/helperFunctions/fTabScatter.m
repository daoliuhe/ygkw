function fTabScatter(Cycles,Table,CatNames,Threshold,Label,LINE,RUN,AxesTab)

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
set(0,'defaultFigurePosition',[spos(3)*.2 spos(4)*.2 spos(3)*.6 spos(4)*.6]);

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

f=figure(1);clf
f.Color = 'w';
f.Colormap = map;

tabgp = uitabgroup(f);
tab = gobjects(1,NTabs);    % tab handle
cb = gobjects(1,NTabs);     % colorbar handles
ax = gobjects(1,NSignals);  % axes handle
p = gobjects(NSignals,Nc);  % fill handle
if SCATTER
    scatHdl = gobjects(1,NSignals);   % scatter plot handle
end
if LINE
    l = gobjects(1,NSignals);   % line handle
    LP = {'Color',.7*[1 1 1],'LineWidth',1};
end
ktab = 0;

extrema = @(x) [min(x,[],1);max(x,[],1)]';
XLim = extrema(Cycles);
YLim = extrema(Table{:,:});
kXcst = find(diff(XLim,[],2)==0);
XLim(kXcst,:) = bsxfun(@plus,XLim(kXcst,:),[-1 1]);
T = [0,Threshold(:)',max([Threshold(end)+100,diff(XLim)])];

kYcst = find(diff(YLim,[],2)==0);
YLim(kYcst,:) = bsxfun(@plus,YLim(kYcst,:),[-1 1]);
for ks =1:NAxesTab:NSignals % ks: signal index
    ktab = ktab + 1;
    kS = ks:min([NSignals,ks+NAxesTab-1]);
    if length(kS)>1
        tabtt = sprintf('Signals %d-%d',ks,kS(end));
    else
        tabtt = sprintf('Signal %d',ks);
    end
    tab(ktab) = uitab(tabgp,'Title',tabtt,'BackGroundColor','w');
    for ka = 1:min([length(kS),NAxesTab])
        ax(kS(ka)) = subplot(AxesTab(1),AxesTab(2),ka,'Parent',tab(ktab));
        for kc = 1:Nc
            p(kS(ka),kc) = patch('XData',XLim(2)-[T(kc)*[1 1] T(kc+1)*[1 1]], ...
                'YData',YLim(kS(ka),[1 2 2 1]), ...
                'FaceColor',map(kc,:),'FaceAlpha',0.2,'EdgeColor','none');
        end
        hold on
        if RUN
            if LINE
                l(kS(ka)) = line(Cycles,nan(NCycles,1),LP{:});
            end
            if SCATTER
                scatHdl(kS(ka)) = scatter(Cycles,nan(NCycles,1),[],Label,'filled');
            end
        else
            if LINE
                l(kS(ka)) = line(Cycles,Table{:,kS(ka)},LP{:});
            end
            if SCATTER
                scatHdl(kS(ka)) = scatter(Cycles,Table{:,kS(ka)},[],Label,'filled');
            end
        end
        hold off
        set(ax(kS(ka)),'XLim',XLim,'YLim',YLim(kS(ka),:))
        title(Table.Properties.VariableNames{kS(ka)})
        xlabel('Cycles')
        set(ax(kS(ka)),'CLim',[1 Nc]);
    end
    cb(ktab)=axes('Units','normalized','Pos',[.925 .125 .025 .75], ...
        'Parent',tab(ktab));
    imagesc((0:Nc-1)','Parent',cb(ktab))
    cb(ktab).XTick = [];
    cb(ktab).YTick = 1:Nc;
    cb(ktab).YTickLabel = CatNames;
    cb(ktab).YTickLabelRotation = 90;
    cb(ktab).YAxisLocation = 'right';
end
shg
if RUN
    for kCycle = 1:NCycles
        for ksignal = 1:NSignals
            if LINE
                l(ksignal).YData(kCycle) = Table{kCycle,ksignal};
            end
            if SCATTER
                scatHdl(ksignal).YData(kCycle) = Table{kCycle,ksignal};
            end
        end
        drawnow limitrate
    end
end
