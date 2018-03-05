function visualise_splitfunc(idx_best,data,dim,t,ig_best,iter) % Draw the split line
r = [-1.5 1.5]; % Data range

subplot(2,2,1);
if (length(dim)==1)
    if dim == 1
        plot([t t],[r(1),r(2)],'r');
    else
        plot([r(1),r(2)],[t t],'r');
    end
else
    if(length(dim)==2)
        y=(-dim(1)*r+t)./dim(2);
        dim='linear';
        plot(r,y,'r', 'LineWidth', 2.5);
        title(sprintf('Linear: IG = %4.2f',ig_best),'FontSize', 20);
    elseif(length(dim)==3)
        str=sprintf('%.3f*y + %.3f*x  + %.3f*x.^2 = %.3f',dim(1),dim(2),dim(3),t);
        dim='quadratic';
        h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
        h.LineWidth=2.5;
        h.LineColor='r';
        title(sprintf('Quad: IG = %4.2f',ig_best),'FontSize', 20);
    elseif(length(dim)==4)
        str=sprintf('%.3f*y + %.3f*x  + %.3f*x.^2 + %.3f*x.^3 = %.3f',dim(1),dim(2),dim(3),dim(4),t);
        dim='quadratic';
        h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
        h.LineWidth=2.5;
        h.LineColor='r';
        title(sprintf('Quad: IG = %4.2f',ig_best),'FontSize', 20);
%         
%         y=(-dim(2)*r-dim(3)*r.^2-dim(4)*r.^3+t)./dim(1);
%         dim='cubic';
%         plot(r,y,'r', 'LineWidth', 2.5);
%         title(sprintf('Cube: IG = %4.2f',ig_best),'FontSize', 20);
    elseif(length(dim)==5)
        str=sprintf('%.3f*x + %.3f*y + %.3f*x.*y + %.3f*x.^2 + %.3f*y.^2 = %.3f',dim(1),dim(2),dim(3),dim(4),dim(5),t);
        dim='quadratic';
        h=ezplot(str,[-1.5 1.5 -1.5 1.5]);
        h.LineWidth=2.5;
        h.LineColor='r';
        title(sprintf('Quad: IG = %4.2f',ig_best),'FontSize', 20);
    
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
subplot(2,2,2);
tmp = hist(data(:,end), unique(data(:,end)));
bar(tmp);
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of parent node');
subplot(2,2,3);
bar(hist(data(idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of left child node');
subplot(2,2,4);
bar(hist(data(~idx_best,end), unique(data(:,end))));
axis([0.5 3.5 0 max(tmp)]);
title('Class histogram of right child node');
hold off;
end