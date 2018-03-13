
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

save('rf_codebook_results_smaller.m', 'build_time', 'accuracy_array');

%%
% Iterate split num and split func
range_splitnum = [2,16,64,128];
range_splitfunc = ["axisaligned", "twopixel"];