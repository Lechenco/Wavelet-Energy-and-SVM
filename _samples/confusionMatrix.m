%defining my colors
f1=[0 0 139]/255;
f4=[50 205 50]/255;
f9=[236 0 0]/255;
f14=[85 26 139]/255;

%example
a=[1 1 0 1 0 0 1 0 1 0 1 1 0 0 1 0 0 0 1 0];
b=[1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 1 0 0 1 0];

figure(1)
plotconfusion(a,b)

%fontsize
set(findobj(gca,'type','text'),'fontsize',20) 

%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)