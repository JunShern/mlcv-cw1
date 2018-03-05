function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 1;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    
    % Split function - Modify here and try other types of split function
%     [idx_, dim, t] = splitAxisAligned(data, D);
%     [idx_, dim, t] = splitLinear(data);
%     [idx_, dim, t] = splitQuadratic(data);
    [idx_, dim, t] = splitCubic(data);
    
    ig = getIG(data,idx_); % Calculate information gain
    
    if visualise
        visualise_splitfunc(idx_,data,dim,t,ig,n);
        pause();
    end
    
    if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
    [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
    end
    
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function [idx_, dim, t] = splitAxisAligned(data, D)
    % Axis-aligned
    dim = randi(D-1); % Pick one random dimension
    d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
    d_max = single(max(data(:,dim))) - eps;
    t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
    idx_ = data(:,dim) < t;
end

% function [idx_, t] = splitLinear(data)
%     a = rand();
%     b = d_min + (d_max-d_mix)*rand();
%     idx_ = (data(:,2) > a*data(:,1) + b); % y > ax + b
% %     line = rand(3,1); % homogeneous line equation L(C0, C1, C2) = C0 + C1x + C2y
% %     homogeneous_coords = data;
% %     homogeneous_coords(:,3) = 1;
% %     t1 = rand();
% %     t2 = -rand();
% %     size(homogeneous_coords)
% %     size(line)
% %     idx_ = (t1 > homogeneous_coords*line) & (homogeneous_coords*line > t2);
% %     t = [line, t1, t2];
% end

%linear learner
function [idx_, dim, t] = splitLinear( data )

    % get maximum and minima along all axis
    axis_max=max(data(:,1:end-1));
    axis_min=min(data(:,1:end-1));
    
    % initialise normal to plane in 2 dimenions
    dim = zeros(size(data(:,1:end-1),2),1); % zeros(2,1)
    
    % generate random points along the 2 axis
    for i=1:size(data(:,1:end-1),2) % 1:2
        dim(i)=axis_min(i)+rand*(axis_max(i)-axis_min(i));
    end
    
    % initialise the threshold
    holder = zeros(1, size(data(:,1:end-1),2)); % zeros(1,2)
    holder(1) = dim(1);
    % determine threshold
    t = holder*dim;
    
    % Pick a random value within the range as threshold
    idx_ = data(:,1:end-1)*dim < t;

end

%Quadratic feature learner
function [idx_, dim, t] = splitQuadratic( data )

    % transform into feature vector 
    non_linear_feat=[data(:,2), data(:,1), data(:,1).^2]; %[data(:,1), data(:,2), data(:,1).*data(:,2), data(:,1).^2, data(:,2).^2];
    
    % get maximum and minima along all axis
    axis_max=max(non_linear_feat);
    axis_min=min(non_linear_feat);
    
    % initialise normal to plane in 3 dimenions
    dim = zeros(size(non_linear_feat,2),1);
    
    % generate random points along the 3 axis
    for i=1:size(non_linear_feat,2)
        dim(i)=axis_min(i)+rand*(axis_max(i)-axis_min(i));
    end
    
    % determine threshold
    t = [dim(1) 0 0]*dim; %[dim(1) 0 0 0 0]*dim;
    
    % Pick a random value within the range as threshold
    idx_ = non_linear_feat*dim < t;

end

%Cubic feature learner
function [idx_, dim, t] = splitCubic( data )
    % transform into feature vector 
    non_linear_feat=[data(:,2), data(:,1), data(:,1).^2, data(:,1).^3];
    %[data(:,1), data(:,2), data(:,1).*data(:,2), data(:,1).^2, data(:,2).^2, (data(:,1).^2).*data(:,2), data(:,1).*(data(:,2).^2), data(:,1).^3, data(:,2).^3];

    % get maximum and minima along all axis
    axis_max=max(non_linear_feat);
    axis_min=min(non_linear_feat);
    
    % initialise normal to plane in 3 dimenions
    dim = zeros(size(non_linear_feat,2),1);
    
    % generate random points along the 3 axis
    for i=1:size(non_linear_feat,2)
        dim(i)=axis_min(i)+rand*(axis_max(i)-axis_min(i));
    end
    
    % determine threshold
    t = [dim(1) 0 0 0]*dim; %[dim(1) 0 0 0 0 0 0 0 0]*dim; 
    
    % Pick a random value within the range as threshold
    idx_ = non_linear_feat*dim < t;

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx,:);
R = data(~idx,:);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,end), unique(X(:,end)));
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end