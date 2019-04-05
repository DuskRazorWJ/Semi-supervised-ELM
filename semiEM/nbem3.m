function [ac,act,label,model] = nbem3(X, init, sx,sy,sn,testx,testy,la,inc)

%  X is the labeled training matrix
%  init is the labeled of X
%  sx and sy is the unlabeled training part
%  sn is the number of split usually 4-16 is ok. Bigger the better but slower.
%  testx and testy is the test data.
%  each colunm is one single data which means X is a d*n matrix.
%  d is deminsional n is the number of data.
%  la and inc are perimeters if you dont know what it is you can leave it


%  written by frank(Jin Sun) from Stevens Institute of Technology
%  franksunjin@gmail.com 
%  cs.stevens.edu/~jsun6
%  Inspired by Michael Chen (sth4nth@gmail.com).
%% initialization
fprintf('Running ... \n');
% Enable this if you have huge memory
% it will let you do parallal
%X=full(X);
[d,n]=size(X);
X=log(X+1);
sx=log(sx+1);
testx=log(testx+1);
R = initialization(X,init);
rR=R;
nc=size(R,2);
%SR = initialization(sx,sy);
SR=zeros(size(sy,2),size(R,2));
%TR = initialization(testx,testy);
TR=zeros(size(testy,2),size(R,2));
ly=[1:n];
Tn=size(sy,2);

[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label));
nly=size(ly,2);
tol = 1e-10;
maxiter = 20;
llh = -inf(1,maxiter);
converged = false;
converged2 = false;
if nargin<8
    la=0.72352-0.14693*log((sn-1)/sn*Tn/n);
    if la<0.1
        la=0.1;
    end
    inc=1;
end
lamda=la
t = 1;
m1=maximization(X,R,[1:n],1,0.08);  %First time prior
[TTR]=expectation(testx,m1, TR, []);
    [~,labelT(1,:)] = max(TTR,[],2);
    r=labelT==testy;
    ac=(sum(r))/(size(r,2));
    fprintf('Test accuracy %f .\n',ac*100);    
  %increase learning rate every step
for i=1:sn
    sxl{i}=sx(:,[i:sn:end]);
    sxu{i}=sx;
    sxu{i}(:,[i:sn:end])=[];
    syl{i}=sy(:,[i:sn:end]);
    syu{i}=sy;
    syu{i}(:,[i:sn:end])=[];
end

for i=1:sn
[RSR]=expectation(sxu{i},m1, SR, []);
[~,rlabel{i}(1,:)] = max(RSR,[],2);
r=rlabel{i}==syu{i};
ac=sum(r)/size(r,2);
%fprintf('accuracy without EM %f .\n',ac*100); 

    t = 1;
converged = false;
R=[rR;RSR];
while ~converged && t < maxiter
    t = t+1;
    %model = maximization([X,sx],R,[1:n],lamda*dec^(i));
    %R = expectation([X,sx],model,R, [1:n]);
    model = maximization([X,sxu{i}],R,[1:n],lamda*inc^(t-1));
    
    [ac,label]=acrate(sxu{i},syu{i},model,size(R,2));

 %   fprintf('semi accuracy %f .\n',ac*100);  
    llh(t)=ac;
    R = initialization([X,sxu{i}],[init,label]);  
    %fprintf('Test accuracy %f .\n',ac*100);  

% I am NOT minimizing the error rate here
% just converge when stable
    converged = abs(llh(t)-llh(t-1))< tol*abs(llh(t));
%   end

end
if converged
    fprintf('Converged in %d steps.\n',t-1);
end

[ac,ll{i}]=acrate(sxl{i},syl{i},model,size(R,2));
%ll{i} = expectation(sxl{i}, model, zeros(size(sxl{i},1),size(R,2)), []);
    fprintf('Semi accuracy %f .\n',ac*100);  
%[ac,~]=acrate(testx,testy,model,size(R,2));
%    fprintf('TEST accuracy %f .\n',ac*100);  
 
    fprintf('Iter %d .\n',i);
end
%train all part together
%first put all part together
sxla=[];
RLA=[];
for i=1:sn
    sxla=[sxla,sxl{i}];
    tmpR=initialization(sxl{i},ll{i},nc);
    RLA=[RLA;tmpR];
end
%[ac,~]=acrate(testx,testy,model,size(R,2));
%    fprintf('before TEST accuracy %f .\n',ac*100);  
model = maximization([X,sxla],[rR;RLA],[1:n],lamda);
    [ac,lable]=acrate(sx,sy,model,size(R,2));
    %fprintf('Final Semi accuracy %f .\n',ac*100);  
[ac,~]=acrate(testx,testy,model,size(R,2));
    %fprintf('Final TEST accuracy %f .\n',ac*100);  

converged = false;
R = initialization([X,sx],[init,lable]);
t=1;  
while ~converged && t < maxiter
    t = t+1;
    %model = maximization([X,sx],R,[1:n],lamda*dec^(i));
    %R = expectation([X,sx],model,R, [1:n]);
    model = maximization([X,sx],R,[1:n],lamda*inc^(t-1));
    
    [ac,label]=acrate(sx,sy,model,size(R,2));

   %fprintf('semi accuracy %f .\n',ac*100);  
    llh(t)=ac;
    R = initialization([X,sx],[init,label]);  
    %fprintf('Test accuracy %f .\n',ac*100);  

% I am NOT minimizing the error rate here
% just converge when stable
    converged = abs(llh(t)-llh(t-1))< tol*abs(llh(t));
%   end

end
    fprintf('semi accuracy %f .\n',ac*100);

   [act,~]=acrate(testx,testy,model,size(R,2));
    fprintf('Test accuracy %f .\n',act*100); 
end

function [ac,label]=acrate(x,y,model,cl)
    R=zeros(size(y,2),cl);
    R=expectation(x,model, R, []);
    [~,label(1,:)] = max(R,[],2);
    r=label==y;
    ac=(sum(r))/(size(r,2));
end

function R = initialization(X, init,nc)
if nargin<3
    nc=max(init);
end
[d,n] = size(X);
if ~(size(init,1) == 1 && size(init,2) == n)
    error('ERROR: init is not valid.');
end
init=[[1:nc],init];
if size(init,1) == 1 && size(init,2) == n+nc  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n+nc,label,1,n+nc,k,n+nc));
    R([1:nc],:)=[];
else
    error('ERROR: init is not valid.');
end
%fprintf('Init finish\n');
end



function R = expectation(X, model, OR, ly)
mu = model.mu;
w = model.weight;
n = size(X,2);
k = size(mu,2);
a=(X'*log(mu)+ones(n,1)*log(w));
%a=-a/median(median(a)); %stable the value
a=-a/mean(mean(a)); 
R = exp(a);
R=R./(sum(R,2)*ones(1,k));
%ensure that the labeled data won't change
R(ly,:)=OR(ly,[1:size(R,2)]);
%fprintf('Expectation finish\n');
end

function model = maximization(X, R,ly,lamda,p)
[d,n] = size(X);
k = size(R,2);
ll=[1:n];
ll(:,ly)=[];
R(ll,:)=R(ll,:)*lamda;
N=1;
if nargin<5
    p=mean(mean(X))*N; 
end
XR=X*R+p;                 %Laplace smooth
nk=sum(XR)+p*k;
w = (sum(R)+p)/(n+k*p);
mu = bsxfun(@times, XR, 1./nk);

model.mu = mu;
model.weight = w;
%fprintf('Maximization finish\n');
end
