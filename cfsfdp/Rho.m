clear all
close all
clc
%选取样本
org=importdata('duichenflame90.mat');

%x需要归一化的真实数据预处理
[N,M]=size(org);
xx=org(:,1:M-1);
%xx=mapminmax(xx);

%生成dist距离矩阵
dist=zeros(N,N);
for i=1:N-1
    for j=i+1:N
        dist(i,j)=norm(xx(i,:)-xx(j,:));
        dist(j,i)=norm(xx(i,:)-xx(j,:));
    end
end
%选取dc
bb=0;
for i=1:N-1
    for j=i+1:N
        bb=bb+1;
        distence(bb)=dist(i,j);
    end
end
percent=0.01;
position=round(bb*percent/100);
sda=sort(distence);
dc=sda(position);

fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);%计算Rho与高斯核的半径

%全0初始化rho向量，这里rho是密度
for i=1:N
  rho(i)=0.;
end

% Gaussian kernel，原型G(x)=e^(-x^2)

for i=1:N-1
  for j=i+1:N
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
  end
end%迭代更新，当i固定，随着j越来越大

maxd=max(max(dist));%整个矩阵的最大值，也就是最大距离

[rho_sorted,ordrho]=sort(rho,'descend');%密度从大到小降序排列，rho_sorted是排序结果向量，ordrho是对应从大到小排列的密度编号
delta(ordrho(1))=-1.;%最大点的delta先设置为-1，其他的为最小距离

nneigh(ordrho(1))=0;%距离最近点的编号

%找密度比自己大，且距离最短的点，delta保存距离，nneigh保存编号
for ii=2:N
   delta(ordrho(ii))=maxd;%每次循环开始，找到最大距离
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))%在比自己密度大的点里找距离最短的
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));%delta取小值
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));%取出距离最大值
disp('Generated file:DECISION GRAPH')%生成决策图文件，输出密度和delta数据
disp('column 1:Density')
disp('column 2:Delta')

fid = fopen('DECISION_GRAPH', 'w');
for i=1:N
   fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
end

disp('Select a rectangle enclosing cluster centers')%选择围绕聚类中心的矩形
scrsz = get(0,'ScreenSize');%获取电脑分辨率，第三个是屏幕宽度，第四个是屏幕高度
figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);%position属性，[left bottom width height]，前两个是原点坐标位置

subplot(2,1,1)%纵向两个图，这是第一个
%画图，并获取句标tt，o型，k：黑色，Marksize：标识符大小，Markfacecolor：标识符填充颜色，markeredgecolor：标识符边缘颜色
tt=plot(rho(:),delta(:),'o','MarkerSize',3,'MarkerFaceColor','k','MarkerEdgeColor','k');
title ('Decision Graph','FontSize',15.0)
xlabel ('\rho')
ylabel ('\delta')


subplot(2,1,1)%第二幅
rect = getrect(1);%从第一幅图中获取矩形矩阵，包含xmin,ymin,widht,height
rhomin=rect(1);%rho为x轴，这里是最小值
deltamin=rect(2);
NCLUST=0;%较类数量
for i=1:N%初始化c1数组
  cl(i)=-1;
end
%聚类中心个数
for i=1:N
    if ((rho(i)>rhomin) && (delta(i)>deltamin))
    %if ( ((rhomin+rect(3))>rho(i)>rhomin) && ((deltamin+rect(4))>delta(i)>deltamin))%统计数据点rho和delta都大于最小值得点，以矩形框得坐下为边界
     NCLUST=NCLUST+1;
     cl(i)=NCLUST;%簇类中心的编号，也就是属于哪一类
     icl(NCLUST)=i;% 逆映射,第 NCLUST 个 cluster 的中心为第 i 号数据点  
  end
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);%群组数
disp('Performing assignation')%执行分配

%%%%%%%%%%%%%%%%%%%%%%%矫正标签%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:NCLUST
%     cl(icl(i))=org(icl(i),M);
% end

%assignation
for i=1:N
  if (cl(ordrho(i))==-1)%=-1说明不是簇类中心
    cl(ordrho(i))=cl(nneigh(ordrho(i)));%则属于离他最近且密度比其大的类，因为rho从最大的顺序赋值，不会出现传递
  end
end
%初始化halo，类光环部分，噪声

if (NCLUST>1)%如果簇类数目大于1 
    for i=1:N
        halo(i)=cl(i);
    end
  for i=1:NCLUST%初始化bord_rho数组
    bord_rho(i)=0.;
  end

  %。。。。。。。。。下面开始机选边界区域。。。。。。。。
  %文中对边界的定义：是簇类点，但是与其他类的数据点的距离小于dc
  %然后在边界找出密度最大的点，密度定义为pd
  %类中大于这个密度的为簇类核心，小于这个密度的为光晕（噪声）
  for i=1:N-1
    for j=i+1:N
        
      if ((cl(i)~=cl(j))&& (dist(i,j)<=dc))%对每个数据点i判断可否成为边界点，如果可以就记录一个平均密度
        rho_aver=(rho(i)+rho(j))/2.;
        if (rho_aver>bord_rho(cl(i))) %看平均密度是否可划分核心和噪声,找最大密度
          bord_rho(cl(i))=rho_aver;
        end
        if (rho_aver>bord_rho(cl(j))) 
          bord_rho(cl(j))=rho_aver;
        end
      end
    end
  end
  for i=1:N
    if (rho(i)<bord_rho(cl(i)))%小于划分密度，就判定为噪声，halo设置为0
      halo(i)=0;
    end
  end
end
for i=1:NCLUST%对每个族类计算
  nc=0;%类中元素个数
  nh=0;%类中核心个数
  for j=1:N
    if (cl(j)==i) %簇类元素个数
      nc=nc+1;
    end
    if (halo(j)==i)%找族类核心 
      nh=nh+1;
    end
  end
  fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end

cmap=colormap;%获取一个图形颜色板，返回一个64*3的矩阵（缺省值情况）
for i=1:NCLUST
   ic=int8((i*64.)/(NCLUST*1.));%ic是颜色设置，选择64的某一行（即只能绘制64类不同的颜色）
   subplot(2,1,1)
   hold on
   plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end


faa = fopen('CLUSTER_ASSIGNATION', 'w');
disp('Generated file:CLUSTER_ASSIGNATION')
disp('column 1:element id')
disp('column 2:cluster assignation without halo control')
disp('column 3:cluster assignation with halo control')
for i=1:N
   fprintf(faa, '%i %i %i\n',i,cl(i),halo(i));
end

%聚类的错误率
%cl1=cl-1;
c3=cl'-org(:,M);
errornum3=sum(c3~=0);
fprintf('Clustering error numbers: %i  \n', errornum3); 
fprintf('Clustering accuracy rate: %f%%  \n', 100-100*errornum3/N); 
aa=[org(:,1:M-1),c3];