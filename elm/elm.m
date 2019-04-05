clear all
close all
clc
%选取样本
org=importdata('IRIS数据集.xls');

[N,M]=size(org);
num02=round(N*0.1);
num08=round(N*0.8);
numtest=N-num08;

%确定类标签是否从0开始
if ismember(0,org(:,M))
    org(:,M)=org(:,M)+1;  
end

%看具体数据集决定*********************是否需要归一化************************
label=org(:,M);
org2=mapminmax(org(:,1:M-1),0,1);
%org=[org2,label];

r=randperm(size(org,1));%1表示行
org1=org(r,:);
xx=org1(:,1:M-1);

train02=org1(1:num02,:);
train08=org1(1:num08,:);
traintest=org1(num08+1:end,:);
traintest01=org1(num02+1:end,:);

%用num02个样本训练elm
train02_fea=train02(:,1:M-1);
train02_lab=train02(:,M);
train02_lab_mat=ind2vec(train02_lab');%

NN=10;
[R,Q]=size(train02_fea);
IW=rand(NN,Q)*2-1;
B=rand(NN,1);
BM1=repmat(B,1,R);
tempH1=IW*train02_fea'+BM1;%
H1=1./(1+exp(-tempH1));
LW1=pinv(H1)'*train02_lab_mat';%


[S,D]=size(traintest01(:,1:M-1)');
BM2=repmat(B,1,D);%400*dd
tempH2=IW*traintest01(:,1:M-1)'+BM2;%400*dd
H2=1./(1+exp(-tempH2));

Y2=(H2'*LW1)';

temp_Y2=zeros(size(Y2));
    for i2 = 1:D
        [max_Y2,index2] = max(Y2(:,i2));%求i列最大值，index表示行
        temp_Y2(index2,i2) = 1;
    end
Y22=vec2ind(temp_Y2);

c2=Y22'-traintest01(:,M);
errornum2=sum(c2~=0);
fprintf('num02elm error numbers: %i  \n', errornum2); 
fprintf('num02elm accuracy rate: %f%%  \n', 100-100*errornum2/D); 

%用num08个样本训练elm
train08_fea=train08(:,1:M-1);
train08_lab=train08(:,M);
train08_lab_mat=ind2vec(train08_lab');%

[r,q]=size(train08_fea);
IW3=rand(NN,q)*2-1;
B3=rand(NN,1);
BM3=repmat(B3,1,r);
tempH3=IW3*train08_fea'+BM3;%
H3=1./(1+exp(-tempH3));
LW3=pinv(H3)'*train08_lab_mat';%

[s,d]=size(traintest(:,1:M-1)');
BM4=repmat(B3,1,d);%400*dd
tempH4=IW3*traintest(:,1:M-1)'+BM4;%400*dd
H4=1./(1+exp(-tempH4));

Y4=(H4'*LW3)';

temp_Y4=zeros(size(Y4));
    for i4 = 1:d
        [max_Y4,index4] = max(Y4(:,i4));%求i列最大值，index表示行
        temp_Y4(index4,i4) = 1;
    end
Y44=vec2ind(temp_Y4);

c8=Y44'-traintest(:,M);
errornum8=sum(c8~=0);
fprintf('num08elm error numbers: %i  \n', errornum8); 
fprintf('num08elm accuracy rate: %f%%  \n', 100-100*errornum8/numtest); 

