clear all
close all
clc

%function [ classfication ] = test( train,test1 )

%load chapter12_wine.mat                       %下载数据
%选取样本
org=importdata('IRIS数据集.xls');

[N,M]=size(org);
num02=round(N*0.1);
num08=round(N*0.8);
numtest=N-num02;

r=randperm(size(org,1));%1表示行
org1=org(r,:);

train=org1(1:num02,1:M-1);
train_group=org1(1:num02,M);
test1=org1(num02+1:N,1:M-1);
test_group=org1(num02+1:N,M);

%train=[wine(1:30,:);wine(60:95,:);wine(131:153,:)]; %选取训练数据
%train_group=[wine_labels(1:30);wine_labels(60:95); wine_labels(131:153)];%选取训练数据类别标识
%test=[wine(31:59,:);wine(96:130,:);wine(154:178,:)];%选取测试数据
%test_group=[wine_labels(31:59);wine_labels(96:130); wine_labels(154:178)]; %选取测试数据类别标识

%数据预处理，用matlab自带的mapminmax将训练集和测试集归一化处理[0,1]之间
[mtrain,ntrain]=size(train);
[mtest,ntest]=size(test1);
dataset=[train;test1];
%[dataset_scale,ps]=mapminmax(dataset',0,1);
dataset_scale=dataset';
dataset_scale=dataset_scale';
train=dataset_scale(1:mtrain,:);
test1=dataset_scale((mtrain+1):(mtrain+mtest),:);

%寻找最优c和g
%粗略选择：c&g 的变化范围是 2^(-10),2^(-9),...,2^(10)
[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-10,10,-10,10);
%精细选择：c 的变化范围是 2^(-2),2^(-1.5),...,2^(4), g 的变化范围是 2^(-4),2^(-3.5),...,2^(4)
%[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-2,4,-4,4,3,0.5,0.5,0.9);

%训练模型
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model=svmtrain(train_group,train,cmd);
disp(cmd);

%测试分类
[predict_label, accuracy, dec_values]=svmpredict(test_group,test1,model);

%打印测试分类结果
%%figure;
%%hold on;
%%plot(test_group,'o');
%%plot(predict_label,'r*');
%%legend('实际测试集分类','预测测试集分类');
%%title('测试集的实际分类和预测分类图','FontSize',10);
%end