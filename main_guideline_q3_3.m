

%% RF Codebook
[data_train, data_test] = getDataRFCodebook('Caltech');

%% RF classifier on RF codebook
param.num = 50;         % Number of trees
param.depth = 5;        % trees depth
param.splitNum = 5;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only
trees = growTrees(data_train,param);

%%
% Test on the dense 2D grid data, and visualise the results ... 
predicted_labels = zeros(size(data_test,1), 1);
p_rf = zeros(size(data_test,1), 10);
for n=1:size(data_test, 1)
    leaves = testTrees([data_test(n,:) 0],trees);
    % average the class distributions of leaf nodes of all trees
    p_rf_sum = sum(trees(1).prob(leaves,:));
    p_rf_mean = p_rf_sum/length(trees);
    p_rf(n,:) = p_rf_mean;
    [~, predicted_labels(n,1)] = max(p_rf_mean);
    [data_test(n,end), predicted_labels(n,1)]
end

accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1)
% compare = [data_test(:,end), predicted_labels(:,1)];