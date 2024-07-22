function [Re, Population] = DimJud(Population, upper, lower,ReAlready)
[N1,N2] =  size(Population);


%% 数据预处理、特征提取
% 稠密度
density = mean(Population~=0,1);
% 计算每维非零数组的统计数据
for i = 1: N2
    dim_Q75(i) = prctile(Population(Population(:,i)~=0,i),75);
    dim_Q50(i) = prctile(Population(Population(:,i)~=0,i),50);
    dim_Q25(i) = prctile(Population(Population(:,i)~=0,i),25);
end
dim_Q75(isnan(dim_Q75)) = 0;
dim_Q50(isnan(dim_Q50)) = 0;
dim_Q25(isnan(dim_Q25)) = 0;
% dim_Q50 = prctile(Population,50);
% dim_Q25 = prctile(Population,25);
% 根据上下界归一化
km_input = zeros(1,N2);
% 大于0的维度，根据上界归一化
upper_index = dim_Q50>0;
km_input(upper_index) = dim_Q75(upper_index)./upper(upper_index);
% 小于0的维度，根据下界归一化
lower_index = dim_Q50<0;
km_input(lower_index) = dim_Q25(lower_index)./lower(lower_index);
% 投影
km_input = sqrt(1-(km_input-1).^2);
% C = kmeans(km_input',2);
% 使用kmeans进行聚类，输入：特征1稠密度，特征2归一化后的数据信息，分为四个类别
[C_indx,C_point] = ExactKmeans([km_input',density'],4);
% 共四个类别，选择三个来更新
% non_activating_index = C_indx == 4;
% 判断两个类别，哪一个类别的的均值大
% [~,index1] = max([mean(km_input(C==1)),mean(km_input(C==2))]);
% index1表示均值大的位置，即non-zero dimension的index
% activating_index = 1-non_activating_index;
% activating_index = C_indx == 1;

[~,a] = max(C_point(:,1)+C_point(:,2));
activating_index = C_indx == a;



%% Case 1: dim_mean > 0
ReNow = false(1,size(ReAlready,2));
upper_index1 = upper_index + activating_index; % 索引：中位数大于0，且聚类
ReNow(upper_index1==2) = true;
% upper_index = dim_Q50>0;
% ReNow(upper_index) = dim_Q75(upper_index)./upper(upper_index)>0.1;
ReNow(ReAlready == true) = false;
range=[];
range =  upper(ReNow) - dim_Q75(ReNow);
if ~isempty(range)
Population(:,ReNow) = rand(N1,size(range,2)).*range + repmat(dim_Q75(ReNow),N1,1);
end
ReAlready(ReNow ==true) = true;

%% Case 2: dim_mean < 0
% lower_index = dim_Q50<0;
% ReNow = false(1,size(ReAlready,2));
upper_index1 =  lower_index + activating_index;
ReNow(upper_index1==2) = true;
% ReNow(lower_index) = dim_Q25(lower_index)./lower(lower_index)>0.1;
ReNow(ReAlready ==true) = false;
range=[];
range =  dim_Q25(ReNow) - lower(ReNow);
if ~isempty(range)
Population(:,ReNow) = -1.*rand(N1,size(range,2)).*range + repmat(dim_Q25(ReNow),N1,1);
end

% get new Re (reinitialization)
ReAlready(ReNow ==true) = true;
Re = ReAlready;
end