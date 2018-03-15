load('kmeans_codebooks.mat');

%% Correctly and incorrectly classified images

load('right_wrong.mat');        %Results with correctly and incorrectly classified images

i = 1;
c = 1;
correct = rightwrong{4,10}(:,1);      %Number of image in the class (15 images in total)
incorrect = rightwrong{4,10}(:,2);      %Number of the class (10 in total)
cnt = 1;

folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes
imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)

    subFolderName = fullfile(folderName,classList{correct(136,1)});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    imgIdx{c} = 1:length(imgList);
    imgIdx_tr = imgIdx{c}(1:imgSel(1));
    imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
    I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));

%subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
    imshow(I);

    if size(I,3) == 3
        I = rgb2gray(I); % PHOW work on gray scale image
    end
%cnt = cnt+1;
%drawnow;


%%

load('right_wrong.mat');
figure('position', [0 0 800 800]);

y = rightwrong{4,10}(:,1);
y_hat = rightwrong{4,10}(:,2);

accuracy_conf = sum(y==y_hat(:,end))/length(y); % Classification accuracy (for Caltech dataset)

%confuse_mat = confusionmat(y, y_hat, 'order', [1:size(y,2)]);
confuse_mat = confusionmat(y, y_hat);
imagesc(confuse_mat);

h = colorbar;
title(sprintf('Confusion Matrix (%.2f %% accuracy)', 100*accuracy_conf));
xlabel('Predicted Class');
ylabel('Actual Class');
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 26);
