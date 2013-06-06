
clear
clc
%%config part
kernel = 'linear';
fileID = fopen(strcat('results_SVM_',kernel,'_noPCA.txt'),'a+');
fileData = fopen(strcat('results_SVMclass_dump.txt'),'a+');
single_value_formatspec = ' %d \n';
double_value_formatspec = ' %d %d \n';
string_formatspec = '%s \n';
%%setup
% First read file
usps = textread('uspsdata.txt');
results = textread('uspscl.txt');

%reshape matrix 
% with 200 rows and 256 columns
usps = reshape(usps,200,256);
%processed dataset
processed_dataset = zeros(1,256);
%%
%preprocessing part begin
for i = 1:200
    shape = usps(i , : );
    shape = reshape(shape,16,16);
    shape = mat2gray(shape);
    %level = graythresh(shape);
    %shape = im2bw(shape,level);
  %  figure,imshow(shape);
    processed_dataset(i , :) = reshape(shape,1,256);
end

%%
%generate random sets - took help from stack overflow%
row_size = size(processed_dataset,1);
random_rows = randperm(row_size); %store this for results
train_dataset = processed_dataset(random_rows(1:100),:);
test_dataset = processed_dataset(random_rows(101:end),:);
results_train = results(random_rows(1:100));
results_test = results(random_rows(101:end));
%for debug only 
%for i = 1:100
%      image_matrix = reshape(train_dataset(i,:),16,16);
%       imwrite(image_matrix,strcat('./images/train_',num2str(i),'_',num2str(results_train(i,1)),'.png'),'png');
%end
%for i = 1:100
%       image_matrix = reshape(test_dataset(i,:),16,16);
%       imwrite(image_matrix,strcat('./images/test_',num2str(i),'_',num2str(results_test(i,1)),'.png'),'png');
%end


%%
%Train svm
SVMstruct = svmtrain(train_dataset,results_train,'Kernel_Function',kernel);
%write SVMstruct to file
SVMclasses = svmclassify(SVMstruct,test_dataset);
%print out SVM classes for double check

fprintf(fileData,single_value_formatspec,SVMclasses);
confusion_matrix = confusionmat(results_test,SVMclasses);

%write confusion matrix to file
disp('Confusion Matrix for SVM ');
confusion_matrix
fprintf(fileID,string_formatspec,'Confusion Matrix for SVM');
fprintf(fileID,double_value_formatspec,confusion_matrix);
fprintf(fileID,string_formatspec,' '); %empty space

%%
%k-means clustering
IDX = kmeans(processed_dataset,2);
%normalize IDX
for i=1:200
    if(IDX(i,1) == 2)
        IDX(i,1) = -1;
    end
        
end
confusion_matrix = confusionmat(results,IDX);
disp('Confusion Matrix for K-Means clustering ');
confusion_matrix
fprintf(fileID,string_formatspec,'Confusion Matrix for K-means - all dataset');
fprintf(fileID,double_value_formatspec,confusion_matrix);
fprintf(fileID,string_formatspec,' ');
fprintf(fileID,string_formatspec,'-------------- ');