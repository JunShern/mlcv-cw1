function visualise_splitfunc(idx_best,data,dim,t,ig_best,iter) % Draw the split line
r = [-1.5 1.5]; % Data range
figure('pos',[100 300 2400 400]);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
subplot(1,4,1);
if (length(dim)==1)
    if dim == 1
        plot([t t],[r(1),r(2)],'r');
        title(sprintf('Axis-aligned: IG = %4.2f',ig_best),'FontSize', 30);
    elseif dim == 2
        plot([r(1),r(2)],[t t],'r');
        title(sprintf('Axis-aligned: IG = %4.2f',ig_best),'FontSize', 30);
    % Two-pixel test
    elseif dim == -1
        plot([r(1)+t,r(2)+t],[r(1),r(2)],'r');
        title(sprintf('Two-pixel: IG = %4.2f',ig_best),'FontSize', 30);
    end
else
    if(length(dim)==2)
        y=(-dim(1)*r+t)./dim(2);
        dim='linear';
        plot(r,y,'r', 'LineWidth', 2.5);
        title(sprintf('Linear: IG = %4.2f',ig_best),'FontSize', 30);
    elseif(length(dim)==3)
        str=sprintf('%.3f*y + %.3f*x  + %.3f*x.^2 = %.3f',dim(1),dim(2),dim(3),t);
        dim='quadratic';
        h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
        h.LineWidth=2.5;
        h.LineColor='r';
        title(sprintf('Quad: IG = %4.2f',ig_best),'FontSize', 30);
    elseif(length(dim)==4)
        str=sprintf('%.3f*y + %.3f*x  + %.3f*x.^2 + %.3f*x.^3 = %.3f',dim(1),dim(2),dim(3),dim(4),t);
        dim='cubic';
        h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
        h.LineWidth=2.5;
        h.LineColor='r';
        title(sprintf('Cubic: IG = %4.2f',ig_best),'FontSize', 30);
%         
%         y=(-dim(2)*r-dim(3)*r.^2-dim(4)*r.^3+t)./dim(1);
%         dim='cubic';
%         plot(r,y,'r', 'LineWidth', 2.5);
%         title(sprintf('Cube: IG = %4.2f',ig_best),'FontSize', 10);
%     elseif(length(dim)==5)
%         str=sprintf('%.3f*x + %.3f*y + %.3f*x.*y + %.3f*x.^2 + %.3f*y.^2 = %.3f',dim(1),dim(2),dim(3),dim(4),dim(5),t);
%         dim='quadratic';
%         h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
%         h.LineWidth=2.5;
%         h.LineColor='r';
%         title(sprintf('Quad: IG = %4.2f',ig_best),'FontSize', 10);
    end
end

hold on;
plot(data(~idx_best,1), data(~idx_best,2), '*', 'MarkerEdgeColor', [.8 .6 .6], 'MarkerSize', 10);
hold on;
plot(data(idx_best,1), data(idx_best,2), '+', 'MarkerEdgeColor', [.6 .6 .8], 'MarkerSize', 10);

hold on;
plot(data(data(:,end)==1,1), data(data(:,end)==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
hold on;
plot(data(data(:,end)==2,1), data(data(:,end)==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
hold on;
plot(data(data(:,end)==3,1), data(data(:,end)==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');

% if ~iter
%     title(sprintf('BEST Split [%i]. IG = %4.2f',dim,ig_best));
% else
%     title(sprintf('Trial %i - Split [%i]. IG = %4.2f',iter,dim,ig_best));
% end
axis([r(1) r(2) r(1) r(2)]);
hold off;

% histogram of base node
subplot(1,4,2);
tmp = hist(data(:,end), unique(data(:,end)));
bar(tmp);
axis([0.5 3.5 0 max(tmp)]);
title('Parent node','FontSize', 30);
subplot(1,4,3);
bar(hist(data(idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Left node','FontSize', 30);
subplot(1,4,4);
bar(hist(data(~idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Right node','FontSize', 30);
hold off;
end

