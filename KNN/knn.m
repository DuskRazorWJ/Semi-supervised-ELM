
%%
clc
clear
close all

%function KNNdatgingTest
%%%选取样本
org=importdata('jain.txt');
[N,M]=size(org);
%数据预处理
r=randperm(size(org,1));%1表示行
data=org(r,:);
%inx=org(0.1*M+1:end,1:M-1);
%labels=org1(1:0.1*M,M);
%data=org1(1:0.1*M,1:M-1);

dataMat = mapminmax(data(:,1:M-1));
labels = data(:,M);
len = round((size(dataMat,1))*0.3);
k = 9;
error = 0;
% 测试数据比例
Ratio = 2./3;
numTest = round(Ratio * len);
% 归一化处理
newdataMat=mapminmax(dataMat);

% 测试
for i = 1:numTest
    classifyresult = KNN(newdataMat(i,:),newdataMat(numTest:len,:),labels(numTest:len,:),k);
    %fprintf('测试结果为：%d  真实结果为：%d\n',[classifyresult labels(i)]);
    if(classifyresult~=labels(i))
        error = error+1;
    end
end
  fprintf('精确度为：%f%% \n',100-100*error/(numTest));
%end

function relustLabel = KNN(inx,data,labels,k)
%   inx 为 输入测试数据，data为样本数据，labels为样本标签
[datarow , datacol] = size(data);%样本的大小
diffMat = repmat(inx,[datarow,1]) - data ;%测数据行重复datarow次
distanceMat = sqrt(sum(diffMat.^2,2));%
[B , IX] = sort(distanceMat,'ascend');
len = min(k,length(B));
relustLabel = mode(labels(IX(1:len)));
end
