clear all
close all
clc
%选取样本
data=importdata('mnist_5_4000.xlsx');

[rows,colum]=size(data);

if ismember(0,data(:,colum))
    data(:,colum)=data(:,colum)+1;
end

num01=round(0.2*rows);
num02=round(0.2*rows);
num08=round(0.8*rows);

randsample=randperm(rows);
data1=data(randsample,:);

X=data1(1:num01,1:colum-1);
init=data1(1:num01,colum)';
sx=data1(num01+1:num08,1:colum-1);
sy=data1(num01+1:num08,colum)';
testx=data1(num08+1:num08+num02,1:colum-1);
testy=data1(num08+1:num08+num02,colum)';

[acc,acct]=nbem2(X,init,sx,sy,testx,testy);



