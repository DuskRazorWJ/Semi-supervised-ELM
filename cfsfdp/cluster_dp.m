clear all
close all
disp('The only input needed is a distance matrix file')
disp('The format of this file should be: ')
disp('Column 1: id of element i')
disp('Column 2: id of element j')
disp('Column 3: dist(i,j)')
mdist=importdata('example_distances.dat');
disp('Reading input distance matrix')
xx=mdist;%xx矩阵是输入的距离数据，格式：点1，点2，点12间的距离
ND=max(xx(:,2));
NL=max(xx(:,1));
if (NL>ND)
  ND=NL;%ND是两列中最大个数，也就是数据的个数
end
N=size(xx,1);%N是xx的行数
%%。。。。。生成距离矩阵dist。。。。。。。。。。。。。。。。。。
for i=1:ND
  for j=1:ND
    dist(i,j)=0;%形成一个全0的ND*ND方阵dist
  end
end
for i=1:N
  ii=xx(i,1); 
  jj=xx(i,2);
  dist(ii,jj)=xx(i,3);%把xx的第三列是距离数据，加到dist矩阵中
  dist(jj,ii)=xx(i,3);%dist是个斜对称矩阵
end
%。。。。。选取截断距离dc。。。。。。。。。。。。。。。。。。
percent=2.0;%取百分之二为截断距离，dc
fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);

position=round(N*percent/100);%返回四舍五入整数值
sda=sort(xx(:,3));%元素重新排序
dc=sda(position);

fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);%计算Rho与高斯核的半径

%全0初始化rho向量，这里rho是密度
for i=1:ND
  rho(i)=0.;
end
%
% Gaussian kernel，原型G(x)=e^(-x^2)
%
for i=1:ND-1
  for j=i+1:ND
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
  end
end%迭代更新，当i固定，随着j越来越大
%
% "Cut off" kernel截断内核
%对于i，计算i与所有j之间距离小于dc的个数
%对于j，以为j比i大，到比对j的时候，前面已经比对过

%？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
%比如i=1,j=2的时候，如果dist(1,2)比截断距离大，那么rho(j)自加
%因为当i=2时，比较对象从j=3开始

%for i=1:ND-1
%  for j=i+1:ND
%    if (dist(i,j)<dc)
%       rho(i)=rho(i)+1.;
%       rho(j)=rho(j)+1.;
%    end
%  end
%end

maxd=max(max(dist));%整个矩阵的最大值，也就是最大距离

[rho_sorted,ordrho]=sort(rho,'descend');%从大到小降序排列，rho_sorted是排序结果向量，ordrho是对应从大到小排列的密度编号
delta(ordrho(1))=-1.;%δ%假如用的截断核，rho是小于截断距离的邻居数，即密度。所以最大点的delta先设置为-1，其他的为最小距离

nneigh(ordrho(1))=0;%距离最近点的编号

%找密度比自己大，且距离最短的点，delta保存距离，nneigh保存编号
for ii=2:ND
   delta(ordrho(ii))=maxd;%每次循环开始，找到最大值
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
for i=1:ND
   fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
end

disp('Select a rectangle enclosing cluster centers')%选择围绕聚类中心的矩形
scrsz = get(0,'ScreenSize');%获取电脑分辨率，第三个是屏幕宽度，第四个是屏幕高度
figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);%position属性，[left bottom width height]，前两个是原点坐标位置
%初始化ind，计算伽马值
for i=1:ND
  ind(i)=i;
  gamma(i)=rho(i)*delta(i);
end
subplot(2,1,1)%纵向两个图，这是第一个
%画图，并获取句标tt，o型，k：黑色，Marksize：标识符大小，Markfacecolor：标识符填充颜色，markeredgecolor：标识符边缘颜色
tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
title ('Decision Graph','FontSize',15.0)
xlabel ('\rho')
ylabel ('\delta')


subplot(2,1,1)%第二幅
rect = getrect(1);%从第一幅图中获取矩形矩阵，包含xmin,ymin,widht,height
rhomin=rect(1);%rho为x轴，这里是最小值
deltamin=rect(2);
NCLUST=0;%较类数量
for i=1:ND%初始化c1数组
  cl(i)=-1;
end
for i=1:ND
  if ( (rho(i)>rhomin) && (delta(i)>deltamin))%统计数据点rho和delta都大于最小值得点，以矩形框得坐下为边界
     NCLUST=NCLUST+1;
     cl(i)=NCLUST;%簇类中心的编号，也就是属于哪一类
     icl(NCLUST)=i;%nclust类的中心编号
  end
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);%群组数
disp('Performing assignation')%执行分配

%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)%=-1说明不是簇类中心
    cl(ordrho(i))=cl(nneigh(ordrho(i)));%则属于离他最近且密度比其大的类，用rho从最大的顺序赋值，不会出现传递
  end
end
%初始化halo，类光环部分，噪声
for i=1:ND
  halo(i)=cl(i);
end
if (NCLUST>1)%如果簇类数目大于1 
  for i=1:NCLUST%初始化bord_rho数组
    bord_rho(i)=0.;
  end
  %。。。。。。。。。下面开始机选边界区域。。。。。。。。
  %文中对边界的定义：是簇类点，但是与其他类的数据点的距离小于dc
  %然后在边界找出密度最大的点，密度定义为pd
  %类中大于这个密度的为簇类核心，小于这个密度的为光晕（噪声）
  for i=1:ND-1
    for j=i+1:ND
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
  for i=1:ND
    if (rho(i)<bord_rho(cl(i)))%小于划分密度，就判定为噪声，halo设置为0
      halo(i)=0;
    end
  end
end
for i=1:NCLUST%对每个族类计算
  nc=0;%类中元素个数
  nh=0;%类中核心个数
  for j=1:ND
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
subplot(2,1,2)%绘制聚类图
disp('Performing 2D nonclassical multidimensional scaling')
Y1 = mdscale(dist, 2, 'criterion','metricstress');%mdscale是非量测多尺度变化
plot(Y1(:,1),Y1(:,2),'o','MarkerSize',2,'MarkerFaceColor','k','MarkerEdgeColor','k');%绘制所有点为黑色点，之后加上去
title ('2D Nonclassical multidimensional scaling','FontSize',15.0)
xlabel ('X')
ylabel ('Y')
for i=1:ND%初始化ND*2的A矩阵
 A(i,1)=0.;
 A(i,2)=0.;
end
for i=1:NCLUST
  nn=0;
  ic=int8((i*64.)/(NCLUST*1.));%对每个簇类确定一个颜色变量ic，即cmap某一行
  for j=1:ND%绘制簇类核心，halo不为0的就是簇类核心，用该颜色绘制
    if (halo(j)==i)
      nn=nn+1;
      A(nn,1)=Y1(j,1);
      A(nn,2)=Y1(j,2);
    end
  end
  hold on
  plot(A(1:nn,1),A(1:nn,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end

%for i=1:ND
%   if (halo(i)>0)
%      ic=int8((halo(i)*64.)/(NCLUST*1.));
%      hold on
%      plot(Y1(i,1),Y1(i,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   end
%end
faa = fopen('CLUSTER_ASSIGNATION', 'w');
disp('Generated file:CLUSTER_ASSIGNATION')
disp('column 1:element id')
disp('column 2:cluster assignation without halo control')
disp('column 3:cluster assignation with halo control')
for i=1:ND
   fprintf(faa, '%i %i %i\n',i,cl(i),halo(i));
end
