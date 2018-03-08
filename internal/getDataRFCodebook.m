function [ data_train, data_query ] = getDataRFCodebook( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        
        data_train_desc = [];
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
                descriptors_with_label = [desc_tr{c,i}; ones(1,size(desc_tr{c,i},2))*c];
                data_train_desc = vertcat(data_train_desc, descriptors_with_label');
                [c, i]
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        data_train_desc_sel = single(vl_colsubset(data_train_desc', 10e4))'; % Randomly select 100k SIFT descriptors for clustering
        
        % RF codebook
        
        % Set the random forest parameters for instance, 
        param.num = 50;         % Number of trees
        param.depth = 7;        % trees depth
        param.splitNum = 5;     % Number of split functions to try
        param.split = 'IG';     % Currently support 'information gain' only
        %%%%%%%%%%%%%%%%%%%%%%
        % Train Random Forest
        % Grow all trees
        trees = growTrees(data_train_desc_sel, param);
        totalLeaves = length(trees(1).prob);
        
        % Vector Quantisation
        % data_train will have 150 rows of <totalLeaves+1> vectors
        data_train = zeros(length(classList)*length(imgIdx_tr), totalLeaves + 1);
        for c = 1:length(classList)
            for i = 1:length(imgIdx_tr)
                % Get nearest cluster centers
                image_data_desc = desc_tr{c,i}';
                leaves = testTrees_fast(image_data_desc, trees);
                % Get histogram codebook
                codebook = histc(reshape(leaves, 1, numel(leaves)), 1:totalLeaves) ./ numel(leaves);
                data_train( (c-1)*length(imgIdx_tr) + i, : ) = [codebook, c];
            end
        end
        
        disp('Encoding Images...')
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Testing image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        data_test_desc = [];
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
                descriptors_with_label = [desc_te{c,i}; ones(1,size(desc_te{c,i},2))*c];
                data_test_desc = vertcat(data_test_desc, descriptors_with_label');
                [c, i]
            end
        end
%         suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end
        
        % Quantisation
        % data_train will have 150 rows of <totalLeaves+1> vectors
        data_query = zeros(length(classList)*length(imgIdx_te), totalLeaves + 1);
        for c = 1:length(classList)
            for i = 1:length(imgIdx_te)
                % Get nearest cluster centers
                image_data_desc = desc_te{c,i}';
                leaves = testTrees_fast(image_data_desc, trees);
                % Get histogram codebook
                codebook = histc(reshape(leaves, 1, numel(leaves)), 1:totalLeaves) ./ numel(leaves);
                data_query( (c-1)*length(imgIdx_te) + i, : ) = [codebook, c];
            end
        end
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
    end
end

