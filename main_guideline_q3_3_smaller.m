
init;

%% RF Codebook
% Iterate num trees and depth
range_num_trees = [5,10,20,40,80];
range_depth = [2,4,6];

build_time = zeros(length(range_depth), length(range_num_trees));
accuracy_array = zeros(length(range_depth), length(range_num_trees));
parfor ind_depth = 1:length(range_depth)
    build_time_row = zeros(1, length(range_num_trees));
    accuracy_array_row = zeros(1, length(range_num_trees));
    for ind_numtrees = 1:length(range_num_trees)
        [range_depth(ind_depth), range_num_trees(ind_numtrees)]
        % Create codebook
        codebook_param = struct(); % Empty the variable for parallel execution
        codebook_param.num = range_num_trees(ind_numtrees);         % Number of trees
        codebook_param.depth = range_depth(ind_depth);        % trees depth
        codebook_param.splitNum = 64;     % Number of split functions to try
        codebook_param.split = 'IG';     % Currently support 'information gain' only
        codebook_param.splitfunc = "axisaligned";
        tic;
        [data_train, data_test] = getDataRFCodebook('Caltech', codebook_param);
        build_time_row(ind_numtrees) = toc;
        
        % Train RF classifier on RF codebook
        classifier_param = struct(); % Empty the variable for parallel execution
        classifier_param.num = 160;         % Number of trees
        classifier_param.depth = 10;        % trees depth
        classifier_param.splitNum = 64;     % Number of split functions to try
        classifier_param.split = 'IG';     % Currently support 'information gain' only
        classifier_param.splitfunc = "axisaligned";
        trees = growTrees(data_train, classifier_param);

        % Predict on the test data, and calculate classification accuracy (%)
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

        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_array_row(ind_numtrees) = accuracy
    end
    build_time(ind_depth,:) = build_time_row;
    accuracy_array(ind_depth,:) = accuracy_array_row
end

save('rf_codebook_results_smaller.mat', 'build_time', 'accuracy_array');

%% PLOTTING FOR VARYING NUM TREES 
load('rf_codebook_results_smaller.mat');
range_num_trees = [5,10,20,40,80];
range_depth = [2,4,6];
figure;
for ind_depth = 1:length(range_depth)
    plot(range_num_trees, accuracy_array(ind_depth,:));
    hold on;
end
title('Accuracy vs Num Trees, varying Depth', 'FontSize', 20);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2','4','6', 'Location', 'southeast');
title(lgd, 'Depth');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_depth = 1:length(range_depth)
    plot(range_num_trees, build_time(ind_depth,:));
    hold on;
end
title('Codebook Time vs Num Trees, varying Depth', 'FontSize', 17);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2','4','6', 'Location', 'southeast');
title(lgd, 'Depth');
grid;
grid minor;
lgd.FontSize = 12;


%% RF Codebook
% Iterate split num and split func
range_splitnum = [10,40,80,120];
range_splitfunc = ["axisaligned"]; %, "twopixel"];

split_build_time = zeros(length(range_splitfunc), length(range_splitnum));
split_accuracy_array = zeros(length(range_splitfunc), length(range_splitnum));
for ind_splitfunc = 1:length(range_splitfunc)
    build_time_row = zeros(1, length(range_splitnum));
    accuracy_array_row = zeros(1, length(range_splitnum));
    for ind_splitnum = 1:length(range_splitnum)
        [range_splitfunc(ind_splitfunc), range_splitnum(ind_splitnum)]
        % Create codebook
        codebook_param = struct(); % Empty the variable for parallel execution
        codebook_param.num = 80;         % Number of trees
        codebook_param.depth = 6;        % trees depth
        codebook_param.splitNum = range_splitnum(ind_splitnum);     % Number of split functions to try
        codebook_param.split = 'IG';     % Currently support 'information gain' only
        codebook_param.splitfunc = range_splitfunc(ind_splitfunc);
        tic;
        [data_train, data_test] = getDataRFCodebook('Caltech', codebook_param);
        build_time_row(ind_splitnum) = toc;
        
        % Train RF classifier on RF codebook
        classifier_param = struct(); % Empty the variable for parallel execution
        classifier_param.num = 160;         % Number of trees
        classifier_param.depth = 10;        % trees depth
        classifier_param.splitNum = 64;     % Number of split functions to try
        classifier_param.split = 'IG';     % Currently support 'information gain' only
        classifier_param.splitfunc = "axisaligned";
        trees = growTrees(data_train, classifier_param);

        % Predict on the test data, and calculate classification accuracy (%)
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

        accuracy = sum(data_test(:,end) == predicted_labels(:,1)) / size(data_test,1);
        accuracy_array_row(ind_splitnum) = accuracy
    end
    split_build_time(ind_splitfunc,:) = build_time_row;
    split_accuracy_array(ind_splitfunc,:) = accuracy_array_row
end

save('rf_codebook_results_split.mat', 'split_build_time', 'split_accuracy_array');

%% PLOTTING FOR VARYING NUM TREES 
load('rf_codebook_results_split.mat');
range_splitfunc = ["axisaligned"]; %, "twopixel"
range_splitnum = [10,20,40,80];
figure;
for ind_splitfunc = 1:length(range_splitfunc)
    plot(range_splitnum, split_accuracy_array(ind_splitfunc,:));
    hold on;
end
title('Accuracy vs Split Number', 'FontSize', 18);
xlabel('Split Number', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis Aligned', 'Location', 'southeast');
title(lgd, 'Split Function');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_splitfunc = 1:length(range_splitfunc)
    plot(range_splitnum, split_build_time(ind_splitfunc,:));
    hold on;
end
title('Codebook Time vs Split Number', 'FontSize', 18);
xlabel('Split Number', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis Aligned', 'Location', 'southeast');
title(lgd, 'Split Function');
grid;
grid minor;
lgd.FontSize = 12;


%%

% Create final optimal codebook
codebook_param = struct(); % Empty the variable for parallel execution
codebook_param.num = 80;         % Number of trees
codebook_param.depth = 6;        % trees depth
codebook_param.splitNum = 80;     % Number of split functions to try
codebook_param.split = 'IG';     % Currently support 'information gain' only
codebook_param.splitfunc = "axisaligned";
[data_train_rf, data_test_rf] = getDataRFCodebook('Caltech', codebook_param);
save('final_rf_codebook.mat', 'data_train_rf', 'data_test_rf');

%% GET DATA FOR VARYING NUM TREES AND DEPTH
load('final_rf_codebook.mat');
range_num_trees = [10, 20, 80, 160];
range_depth = [2,5,8,10];

accuracy_array = zeros(length(range_depth), length(range_num_trees));
train_time_rf_trees = zeros(length(range_depth), length(range_num_trees));
test_time_rf_trees = zeros(length(range_depth), length(range_num_trees));
for ind_depth = 1:length(range_depth)
    build_time_row = zeros(1, length(range_num_trees));
    accuracy_array_row = zeros(1, length(range_num_trees));
    for ind_numtrees = 1:length(range_num_trees)
        [range_depth(ind_depth), range_num_trees(ind_numtrees)]
        
        % Train RF classifier on RF codebook
        tic;
        classifier_param = struct(); % Empty the variable for parallel execution
        classifier_param.num = range_num_trees(ind_numtrees);         % Number of trees
        classifier_param.depth = range_depth(ind_depth);        % trees depth
        classifier_param.splitNum = 64;     % Number of split functions to try
        classifier_param.split = 'IG';     % Currently support 'information gain' only
        classifier_param.splitfunc = "axisaligned";
        trees = growTrees(data_train_rf, classifier_param);
        train_time_rf_trees(ind_depth, ind_numtrees) = toc;
        
        % Predict on the test data, and calculate classification accuracy (%)
        tic;
        predicted_labels = zeros(size(data_test_rf,1), 1);
        p_rf = zeros(size(data_test_rf,1), 10);
        for n=1:size(data_test_rf, 1)
            leaves = testTrees([data_test_rf(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            [data_test_rf(n,end), predicted_labels(n,1)]
        end
        test_time_rf_trees(ind_depth, ind_numtrees) = toc;

        accuracy = sum(data_test_rf(:,end) == predicted_labels(:,1)) / size(data_test_rf,1);
        accuracy_array_row(ind_numtrees) = accuracy
    end
    accuracy_array(ind_depth,:) = accuracy_array_row
end

save('rf_classifier_results.mat', 'accuracy_array', 'train_time_rf_trees', 'test_time_rf_trees');

%% PLOTTING FOR VARYING NUM TREES 
load('rf_classifier_results.mat');
range_num_trees = [10, 20, 80, 160];
range_depth = [2,5,8,10];
figure;
for ind_depth = 1:length(range_depth)
    plot(range_num_trees, accuracy_array(ind_depth,:));
    hold on;
end
title('Accuracy vs Num Trees, varying Depth', 'FontSize', 20);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2','5','8','10', 'Location', 'southeast');
title(lgd, 'Depth');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_depth = 1:length(range_depth)
    plot(range_num_trees, train_time_rf_trees(ind_depth,:));
    hold on;
end
title('Train Time vs Num Trees, varying Depth', 'FontSize', 17);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2','5','8','10', 'Location', 'southeast');
title(lgd, 'Depth');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_depth = 1:length(range_depth)
    plot(range_num_trees, test_time_rf_trees(ind_depth,:));
    hold on;
end
title('Test Time vs Num Trees, varying Depth', 'FontSize', 17);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('2','5','8','10', 'Location', 'southeast');
title(lgd, 'Depth');
grid;
grid minor;
lgd.FontSize = 12;


%% GET DATA FOR VARYING NUM TREES AND DEPTH
load('final_rf_codebook.mat');
range_splitnum = [10, 20, 80, 160];
range_splitfunc = ["axisaligned", "twopixel"];

accuracy_array_split = zeros(length(range_splitfunc), length(range_splitnum));
train_time_rf_split = zeros(length(range_splitfunc), length(range_splitnum));
test_time_rf_split = zeros(length(range_splitfunc), length(range_splitnum));
for ind_splitfunc = 1:length(range_splitfunc)
    accuracy_array_row = zeros(1, length(range_splitnum));
    for ind_splitnum = 1:length(range_splitnum)
        [range_splitfunc(ind_splitfunc), range_splitnum(ind_splitnum)]
        
        % Train RF classifier on RF codebook
        tic;
        classifier_param = struct(); % Empty the variable for parallel execution
        classifier_param.num = 160;         % Number of trees
        classifier_param.depth = 8;        % trees depth
        classifier_param.splitNum = range_splitnum(ind_splitnum);     % Number of split functions to try
        classifier_param.split = 'IG';     % Currently support 'information gain' only
        classifier_param.splitfunc = range_splitfunc(ind_splitfunc);
        trees = growTrees(data_train_rf, classifier_param);
        train_time_rf_split(ind_splitfunc, ind_splitnum) = toc;
        
        % Predict on the test data, and calculate classification accuracy (%)
        tic;
        predicted_labels = zeros(size(data_test_rf,1), 1);
        p_rf = zeros(size(data_test_rf,1), 10);
        for n=1:size(data_test_rf, 1)
            leaves = testTrees([data_test_rf(n,:) 0],trees);
            % average the class distributions of leaf nodes of all trees
            p_rf_sum = sum(trees(1).prob(leaves,:));
            p_rf_mean = p_rf_sum/length(trees);
            p_rf(n,:) = p_rf_mean;
            [~, predicted_labels(n,1)] = max(p_rf_mean);
            [data_test_rf(n,end), predicted_labels(n,1)]
        end
        test_time_rf_split(ind_splitfunc, ind_splitnum) = toc;

        accuracy = sum(data_test_rf(:,end) == predicted_labels(:,1)) / size(data_test_rf,1);
        accuracy_array_row(ind_splitnum) = accuracy
    end
    accuracy_array_split(ind_splitfunc,:) = accuracy_array_row
end

save('rf_classifier_results_split.mat', 'accuracy_array_split', 'train_time_rf_split', 'test_time_rf_split');

%% PLOTTING FOR VARYING NUM TREES 
load('rf_classifier_results.mat');
range_splitnum = [10, 20, 80, 160];
range_splitfunc = ["axisaligned", "twopixel"];
figure;
for ind_splitfunc = 1:length(range_splitfunc)
    plot(range_splitnum, accuracy_array(ind_splitfunc,:));
    hold on;
end
title('Accuracy vs Split Num, varying Split Func', 'FontSize', 18);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split Function');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_splitfunc = 1:length(range_splitfunc)
    plot(range_splitnum, train_time_rf_split(ind_splitfunc,:));
    hold on;
end
title('Train Time vs Split Num, varying Split Func', 'FontSize', 18);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split Function');
grid;
grid minor;
lgd.FontSize = 12;

figure;
for ind_splitfunc = 1:length(range_splitfunc)
    plot(range_splitnum, test_time_rf_split(ind_splitfunc,:));
    hold on;
end
title('Test Time vs Split Num, varying Split Func', 'FontSize', 18);
xlabel('Number of Trees', 'FontSize', 15);
ylabel('Accuracy (0-1)', 'FontSize', 15);
lgd = legend('Axis-aligned', 'Two-pixel', 'Location', 'southeast');
title(lgd, 'Split Function');
grid;
grid minor;
lgd.FontSize = 12;
