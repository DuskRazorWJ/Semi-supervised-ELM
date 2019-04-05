% Semi-supervised ELM (US-ELM) for semi-supervised classification.
% Ref: Huang Gao, Song Shiji, Gupta JND, Wu Cheng, Semi-supervised and
% unsupervised extreme learning machines, IEEE Transactions on Cybernetics, 2014
clc;
format compact;
clear; 

addpath(genpath('functions'))

% load data
trial=1; 
data=importdata('jain.txt');
[rows,cols]=size(data);
%data(:,1:cols-1)=mapminmax(data(:,1:cols-1));
num01=round(rows*0.1);
num06=round(rows*0.3);
numtest=rows-2*num01-num06;


if ismember(0,data(:,cols))
    data(:,cols)=data(:,cols)+1;  
end
%%%%%%%¶þ·ÖÀà%%%%%%%%%%%%%%%%%%
for i=1:rows
    if data(i,cols)==2
        data(i,cols)=-1;
    end
end

r=randperm(rows);
data=data(r,:);

X=data(:,1:cols-1);
y=data(:,cols);
if ismember(y,0)
    y=y+1;
end

Xl=X(1:num01,:);
Yl=y(1:num01,:);

Xv=X(num01:num01+num01,:);
Yv=y(num01:num01+num01,:);

Xu=X(num01+num01:num01+num01+num06,:);
Yu=y(num01+num01:num01+num01+num06,:);

Xt=X(num01+num01+num06+1:end,:);
Yt=y(num01+num01+num06+1:end,:);
%%%%%%%%%%%%%% train ss-elm
% Note that manifold regualarization are sensitive to the hyperparameters of graph Laplacian

% Compute graph Laplacian
options.NN=50;
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';

options.LaplacianNormalize=1;
options.LaplacianDegree=5;
L=laplacian(options,[Xl;Xu]);

paras.NumHiddenNeuron=10;
paras.NoDisplay=1;
paras.Kernel='sigmoid';

% model selection using the validation set
acc_v=zeros(10,10);
acc_test=zeros(10,10);
acc_max=0;
for i=1:10
    paras.C=10^(i-5);
    for j=1:10
        paras.lambda=10^(7-j);
        elmModel=sselm(Xl,Yl,Xu,L,paras);
        [acc_v(i,j),MSE(i,j),~,~]=sselm_predict(Xv,Yv,elmModel);
        [acc_test(i,j),~,~]=sselm_predict(Xt,Yt,elmModel);       
        if acc_v(i,j)>acc_max
            acc_max=acc_v(i,j);
            elmModel_best=elmModel;
        end
    end
end

%[acc_tmp,~,~]=sselm_predict(Xu,Yu,elmModel_best);
%accu_u(trial)=acc_tmp
%[acc_tmp,~,~]=sselm_predict(Xv,Yv,elmModel_best);
%accu_v(trial)=acc_tmp
[acc_tmp,~,~,predict]=sselm_predict(Xt,Yt,elmModel_best);
accu_t(trial)=acc_tmp