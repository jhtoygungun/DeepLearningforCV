function Igo = GetRemoveResult(f, e)
% 图像并联去噪函数。
% 将输入的权值向量，串联结果，通过加权求和的方式进行处理。
Igo = f.df1/f.df*double(e.eroded_co12)+f.df2/f.df*double(e.eroded_co22)+...
    f.df3/f.df*double(e.eroded_co32)+f.df4/f.df*double(e.eroded_co42);
Igo = mat2gray(Igo);