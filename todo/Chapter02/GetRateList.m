function f = GetRateList(Ig, e)
% 图像权值计算函数，根据串联结果与原始图像的差异程度进行计算

f.df1 = sum(sum(abs(double(e.eroded_co12)-double(Ig))));
f.df2 = sum(sum(abs(double(e.eroded_co22)-double(Ig))));
f.df3 = sum(sum(abs(double(e.eroded_co32)-double(Ig))));
f.df4 = sum(sum(abs(double(e.eroded_co42)-double(Ig))));
f.df = sum([f.df1 f.df2 f.df3 f.df4]);