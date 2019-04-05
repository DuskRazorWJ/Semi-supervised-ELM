function [ac, act] = nbem2(X, init, sx,sy,testx,testy,la,inc)
%  X is the labeled training matrix
%  init is the labeled of X
%  sx and sy is the unlabeled training part
%  testx and testy is the test data.
%  each colunm is one single data which means X is a d*n matrix.
%  d is deminsional n is the number of data.
%  la and inc are perimeters if you dont know what it is you can leave it
%  written by frank(Jin Sun) from Stevens Institute of Technology
%  franksunjin@gmail.com 
%  cs.stevens.edu/~jsun6
%  Inspired by Michael Chen (sth4nth@gmail.com).
%% initialization
[d,n]=size(X);
X=log(X+1);
sx=log(sx+1);
testx=log(testx+1);
NC=max([init,sy,testy]);
R = initialization(X,init,NC);%the same as 'ind2vec(X)'
k=size(R,2);%The number of categories
[~,labelr(1,:)] = max(R,[],2);
R = R(:,unique(labelr));
tol = 1e-15;
maxiter = 10;
llh = -inf(1,maxiter);
converged = false;
if nargin<7
    la=0.72352-0.14693*log(size(sy,2)/n);
    if la<0.1
        la=0.1;
    end
    inc=1;
end
lamda=la;
t = 1;
%m1=maximization(X,R,[1:size(X,2)],1,0.08); %prior for first
m1=maximization(X,R,[1:size(X,2)],lamda,0.08);
[RR]=expectation(sx',m1,zeros(size(sy,2),k), []);
[~,label(1,:)] = max(RR,[],2);
r=label==sy;
ac=sum(r)/size(r,2);
fprintf('accuracy without EM %f .\n',ac*100);
TR = initialization(testx,testy,NC);
[TTR]=expectation(testx',m1, TR, []);
    [~,labelT(1,:)] = max(TTR,[],2);
    r=labelT==testy;
    ac=(sum(r))/(size(r,2));
    fprintf('Test accuracy %f .\n',ac*100);    
    %increase learning rate every step
  
R=[R;RR]; 
%R=initialization(X,label);
while ~converged && t < maxiter
    t = t+1;
    model = maximization([X',sx']',R,[1:n],lamda*inc^(t-1));
    [ac,label]=acrate(sx,sy,model,size(R,2));
    %semi accuracy
    %fprintf('semi accuracy %f .\n',ac*100);  
    llh(t)=ac;
    R = initialization([X',sx']',[init,label],NC);  
% I am NOT minimizing the error rate here
% just converge when stable
    converged = abs(llh(t)-llh(t-1))< tol*abs(llh(t));
end
fprintf('Semi accuracy %f .\n',ac*100);  
llh = llh(2:t);
if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end
     [act,label]=acrate(testx,testy,model,size(R,2));
    fprintf('Final TEST accuracy %f .\n',act*100);  
end
function [ac,label]=acrate(x,y,model,cl)
    R=zeros(size(y,2),cl);
    R=expectation(x',model, R, []);
    [~,label(1,:)] = max(R,[],2);
    r=label==y;
    ac=(sum(r))/(size(r,2));
end
function R = initialization(X, init,NC)%the same as ind2vec
[n,d] = size(X);
if ~(size(init,1) == 1 && size(init,2) == n)
    error('ERROR: init is not valid.');
end
if isstruct(init)  % initialize with a model
    R  = expectation(X,init);
elseif length(init) == 1  % random initialization
    k = init;
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
    while k ~= length(u)
        idx = randsample(n,k);
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        [u,~,label] = unique(label);
    end
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == d  %initialize with only centers
    k = size(init,2);
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end
%fprintf('Init finish\n');
if (k<NC)
    R=[R,zeros(n,NC-k)];
end
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
XR=X'*R+p;                 %Laplace smooth
nk=sum(XR)+p*k;
w = (sum(R)+p)/(n+k*p);
mu = bsxfun(@times, XR, 1./nk);

model.mu = mu;
model.weight = w;
%fprintf('Maximization finish\n');
end