%% Q3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% experiment with Caltech101 dataset for image categorisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

%%
% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
k_range = 2.^linspace(1,10,10);
ind = 1;
for k = k_range
    k % Print iteration
    [data_train_not_norm, data_test_not_norm] = getData('Caltech', k);
    
    % Normalize histograms to be between 0 and 1
    data_train = data_train_not_norm;
    data_test = data_test_not_norm;
    for i = 1:size(data_train, 1)
        data_train(i,1:end-1) = data_train_not_norm(i,1:end-1) / sum(data_train_not_norm(i,1:end-1));
    end
    for i = 1:size(data_test, 1)
        data_test(i,1:end-1) = data_test_not_norm(i,1:end-1) / sum(data_test_not_norm(i,1:end-1));
    end
    
    data_train_arr{ind} = data_train;
    data_test_arr{ind} = data_test;
    ind = ind + 1;
end
save('kmeans_codebooks.mat','data_train_arr','data_test_arr');

%%
% Visualize histograms
k_ind = 8; %3; % Change this for different histogram sizes
figure;
for i = 1:size(data_train_arr{k_ind}, 1)
    dat = data_train_arr{k_ind};
    subplot(2,1,1);
    bar(data_train_arr{k_ind}(i,1:end-1));
    xlabel("K-means codewords", 'FontSize', 15);
    ylabel("Normalized frequency");
    title(sprintf('BoW Histogram (Image Class %d, K = %d)', dat(i, end), size(dat,2)-1 ), 'FontSize', 18);
    
    subplot(2,1,2);
    bar(dat(i+1,1:end-1));
    xlabel("K-means codewords", 'FontSize', 15);
    ylabel("Normalized frequency");
    
    title(sprintf('BoW Histogram (Image Class %d, K = %d)', dat(i+1, end), size(dat,2)-1 ), 'FontSize', 18);
    pause();
    
end
close all;

%% GET DATA FOR VARYING NUM TREES
load('kmeans_codebooks.mat');
k_range = 2.^linspace(1,10,10);
num_trees_range = [10, 20, 80, 160];
accuracy_results_numtrees = zeros(length(num_trees_range), length(k_range));
train_time = zeros(length(num_trees_range), length(k_range));
test_time = zeros(length(num_trees_range), length(k_range));
for t_ind = 1:length(num_trees_range)
    param.num = num_trees_range(t_ind);         % Number of trees
    param.depth = 5;        % trees depth
    param.splitNum = 3;     % Number of split functions to try
    param.split = 'IG';     % Currently support 'information gain' only
    param.splitfunc = "axisaligned";
    for k_ind = 1:length(k_range)
        param.num, k_range(k_ind) % Print iteration
        data_train = data_train_arr{k_ind};
        data_test = data_test_arr{k_ind};

        %%%%%%%%%%%%%%%%%%%%%%
        % Train Random Forest

        % Grow all trees
        tic;
        trees = growTrees(data_train,param);
        train_time_numtrees(t_ind,k_ind) = toc;
        trees_array{k_ind} = trees;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % random forest codebook for Caltech101 image categorisation
        % .....

        % Test on the dense 2D grid data, and visualise the results ... 
        predicted_labels = zeros(size(data_test,1), 1);
        p_rf = zeros(size(data_test,1), 10);
        tic;
        for n=1:size(data_test, 1)
            leaves = testTrees([data_test(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            %[data_test(n,end), predicted_labels(n,1)];
        end
        test_time_numtrees(t_ind,k_ind) = toc;
        
        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_results_numtrees(t_ind,k_ind) = accuracy
        % v = horzcat(data_test(:,end), predicted_labels(:,1), data_test(:,end) == predicted_labels(:,1));
    end
end
save('3_kmeans_numtrees.mat','accuracy_results_numtrees','train_time_numtrees','test_time_numtrees');

%% PLOTTING FOR VARYING NUM TREES 
load('3_kmeans_numtrees.mat');
k_range = 2.^linspace(1,10,10);
num_trees_range = [10, 20, 80, 160];
figure;
for tree_count = 1:length(num_trees_range)
    plot(k_range, accuracy_results_numtrees(tree_count,:));
    hold on;
end
title('Accuracy vs k, varying Num Trees', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('10','20','80','160', 'Location', 'southeast');
title(lgd, 'Num trees');
lgd.FontSize = 12;

figure;
for tree_count = 1:length(num_trees_range)
    plot(k_range, train_time_numtrees(tree_count,:));
    hold on;
end
title('Train Time vs k, varying Num Trees', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('10','20','80','160', 'Location', 'southeast');
title(lgd, 'Num trees');
lgd.FontSize = 12;

figure;
for tree_count = 1:length(num_trees_range)
    plot(k_range, test_time_numtrees(tree_count,:));
    hold on;
end
title('Test Time vs k, varying Num Trees', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('10','20','80','160', 'Location', 'southeast');
title(lgd, 'Num trees');
lgd.FontSize = 12;


%% GET DATA FOR VARYING DEPTH
load('kmeans_codebooks.mat');
k_range = 2.^linspace(5,10,10);
depth_range = [2, 5, 10, 12];
accuracy_results_depth = zeros(length(depth_range), length(k_range));
train_time_depth = zeros(length(depth_range), length(k_range));
test_time_depth = zeros(length(depth_range), length(k_range));
for t_ind = 1:length(depth_range)
    param.num = 160;         % Number of trees
    param.depth = depth_range(t_ind);        % trees depth
    param.splitNum = 3;     % Number of split functions to try
    param.split = 'IG';     % Currently support 'information gain' only

    for k_ind = 1:length(k_range)
        param.depth, k_range(k_ind) % Print iteration
        data_train = data_train_arr{k_ind};
        data_test = data_test_arr{k_ind};

        %%%%%%%%%%%%%%%%%%%%%%
        % Train Random Forest

        % Grow all trees
        tic;
        trees = growTrees(data_train,param);
        train_time_depth(t_ind,k_ind) = toc;
%         trees_array{k_ind} = trees;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % random forest codebook for Caltech101 image categorisation
        % .....

        % Test on the dense 2D grid data, and visualise the results ... 
        predicted_labels = zeros(size(data_test,1), 1);
        p_rf = zeros(size(data_test,1), 10);
        tic;
        for n=1:size(data_test, 1)
            leaves = testTrees([data_test(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            %[data_test(n,end), predicted_labels(n,1)];
        end
        test_time_depth(t_ind,k_ind) = toc;
        
        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_results_depth(t_ind,k_ind) = accuracy;
        % v = horzcat(data_test(:,end), predicted_labels(:,1), data_test(:,end) == predicted_labels(:,1));
    end
end

save('3_kmeans_depth.mat','accuracy_results_depth','train_time_depth','test_time_depth');

%% PLOTTING FOR VARYING DEPTH
load('3_kmeans_depth.mat');
k_range = 2.^linspace(5,10,10);
depth_range = [2, 5, 10, 12];
figure;
for count = 1:length(depth_range)
    plot(k_range, accuracy_results_depth(count,:));
    hold on;
end
title('Accuracy vs k, varying Depth', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2', '5', '10', '12', 'Location', 'southeast');
title(lgd, 'Tree depth');
lgd.FontSize = 12;

figure;
for count = 1:length(depth_range)
    plot(k_range, train_time_depth(count,:));
    hold on;
end
title('Train Time vs k, varying Tree Depth', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('2', '5', '10', '12', 'Location', 'southeast');
title(lgd, 'Tree depth');
lgd.FontSize = 12;

figure;
for count = 1:length(depth_range)
    plot(k_range, test_time_depth(count,:));
    hold on;
end
title('Test Time vs k, varying Tree Depth', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('2', '5', '10', '12', 'Location', 'southeast');
title(lgd, 'Tree depth');
lgd.FontSize = 12;


%% GET DATA FOR VARYING SPLITNUM
load('kmeans_codebooks.mat');
k_range = 2.^linspace(5,10,10);
splitnum_range = [2, 16, 64, 128];
accuracy_results_splitnum = zeros(length(splitnum_range), length(k_range));
train_time_splitnum = zeros(length(splitnum_range), length(k_range));
test_time_splitnum = zeros(length(splitnum_range), length(k_range));
for t_ind = 1:length(splitnum_range)
    param.num = 160;         % Number of trees
    param.depth = 10;        % trees depth
    param.splitNum = splitnum_range(t_ind);     % Number of split functions to try
    param.split = 'IG';     % Currently support 'information gain' only

    for k_ind = 1:length(k_range)
        param.depth, k_range(k_ind) % Print iteration
        data_train = data_train_arr{k_ind};
        data_test = data_test_arr{k_ind};

        %%%%%%%%%%%%%%%%%%%%%%
        % Train Random Forest

        % Grow all trees
        tic;
        trees = growTrees(data_train,param);
        train_time_splitnum(t_ind,k_ind) = toc;
%         trees_array{k_ind} = trees;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % random forest codebook for Caltech101 image categorisation
        % .....

        % Test on the dense 2D grid data, and visualise the results ... 
        predicted_labels = zeros(size(data_test,1), 1);
        p_rf = zeros(size(data_test,1), 10);
        tic;
        for n=1:size(data_test, 1)
            leaves = testTrees([data_test(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            %[data_test(n,end), predicted_labels(n,1)];
        end
        test_time_splitnum(t_ind,k_ind) = toc;
        
        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_results_splitnum(t_ind,k_ind) = accuracy;
        % v = horzcat(data_test(:,end), predicted_labels(:,1), data_test(:,end) == predicted_labels(:,1));
    end
end
save('3_kmeans_splitnum.mat','accuracy_results_splitnum','train_time_splitnum','test_time_splitnum');

%% PLOTTING FOR VARYING SPLITNUM
load('3_kmeans_splitnum.mat');
k_range = 2.^linspace(5,10,10);
splitnum_range = [2, 16, 64, 128];
figure;
for count = 1:length(splitnum_range)
    plot(k_range, accuracy_results_splitnum(count,:));
    hold on;
end
title('Accuracy vs k, varying Split Num', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2', '16', '64', '128', 'Location', 'southeast');
title(lgd, 'Split num');
lgd.FontSize = 12;

figure;
for count = 1:length(splitnum_range)
    plot(k_range, train_time_splitnum(count,:));
    hold on;
end
title('Train Time vs k, varying Split Num', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('2', '16', '64', '128', 'Location', 'southeast');
title(lgd, 'Split num');
lgd.FontSize = 12;

figure;
for count = 1:length(splitnum_range)
    plot(k_range, test_time_splitnum(count,:));
    hold on;
end
title('Test Time vs k, varying Split Num', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('2', '16', '64', '128', 'Location', 'southeast');
title(lgd, 'Split num');
lgd.FontSize = 12;


%% GET DATA FOR VARYING SPLIT FUNC
load('kmeans_codebooks.mat');
k_range = 2.^linspace(5,10,10);
splitfunc_range = ["axisaligned", "twopixel"];
accuracy_results_splitfunc = zeros(length(splitfunc_range), length(k_range));
train_time_splitfunc = zeros(length(splitfunc_range), length(k_range));
test_time_splitfunc = zeros(length(splitfunc_range), length(k_range));

for t_ind = 1:length(splitfunc_range)
    param.num = 160;         % Number of trees
    param.depth = 10;        % trees depth
    param.splitNum =  128;     % Number of split functions to try
    param.split = 'IG';     % Currently support 'information gain' only
    param.splitfunc = splitfunc_range(:,t_ind);

    for k_ind = 1:length(k_range)
        splitfunc_range(t_ind), k_range(k_ind) % Print iteration
        data_train = data_train_arr{k_ind};
        data_test = data_test_arr{k_ind};

        %%%%%%%%%%%%%%%%%%%%%%
        % Train Random Forest

        % Grow all trees
        tic;
        trees = growTrees(data_train,param);
        train_time_splitfunc(t_ind,k_ind) = toc;
    %         trees_array{k_ind} = trees;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % random forest codebook for Caltech101 image categorisation
        % .....

        % Test on the dense 2D grid data, and visualise the results ... 
        predicted_labels = zeros(size(data_test,1), 1);
        p_rf = zeros(size(data_test,1), 10);
        tic;
        for n=1:size(data_test, 1)
            leaves = testTrees([data_test(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            %[data_test(n,end), predicted_labels(n,1)];
        end
        test_time_splitfunc(t_ind,k_ind) = toc;

        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_results_splitfunc(t_ind,k_ind) = accuracy
        % v = horzcat(data_test(:,end), predicted_labels(:,1), data_test(:,end) == predicted_labels(:,1));
    end
end
save('3_kmeans_splitfunc.mat','accuracy_results_splitfunc','train_time_splitfunc','test_time_splitfunc');

%% PLOTTING FOR VARYING SPLIT FUNC
load('3_kmeans_splitfunc.mat');
k_range = 2.^linspace(5,10,10);
splitfunc_range = ["axisaligned", "twopixel"];
figure;
for count = 1:length(splitfunc_range)
    plot(k_range, accuracy_results_splitfunc(count,:));
    hold on;
end
title('Accuracy vs k, varying Split Function', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split func');
lgd.FontSize = 12;

figure;
for count = 1:length(splitfunc_range)
    plot(k_range, train_time_splitfunc(count,:));
    hold on;
end
title('Train Time vs k, varying Split Function', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split func');
lgd.FontSize = 12;

figure;
for count = 1:length(splitfunc_range)
    plot(k_range, test_time_splitfunc(count,:));
    hold on;
end
title('Test Time vs k, varying Split Function', 'FontSize', 20);
xlabel('Codebook size (Number of Clusters, k)', 'FontSize', 15);
ylabel('Time (s)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split func');
lgd.FontSize = 12;

% 
% %%
% % [~,c] = max(p_rf');
% [~,c] = max(p_rf');
% accuracy_rf = sum(c==data_test(:,end)')/length(c); % Classification accuracy (for Caltech dataset)
% idx = sub2ind([10, 10], data_test(:,end)', c) ;
% conf = zeros(10) ;
% conf = vl_binsum(conf, ones(size(idx)), idx) ;
% 
% imagesc(conf) ;
% title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * accuracy_rf) ) ;