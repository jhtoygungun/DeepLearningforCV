clc; clear all; close all;

%% 输入图像并进行加噪声处理
filename = fullfile(pwd,'images/im.jpg');
% f = fullfile(filepart1,...,filepartN) 根据指定的文件夹和文件名构建完整的文件设定。fullfile 在必要情况下插入依平台而定的文件分隔符，但不添加尾随的文件分隔符。在 Windows® 平台上，文件分隔符为反斜杠 (\)。在其他平台上，文件分隔符可能为不同字符。
% pwd : 获取当前文件的目录地址
Img = imread(filename);

display(ndims(Img));

if ndims(Img) == 3
    I = rgb2gray(Img);
else
    I = Img;

end

Ig = imnoise(I, 'poisson');
% J = imnoise(I,'gaussian') 将方差为 0.01 的零均值高斯白噪声添加到灰度图像 I。

%% 去噪部分

% 获取算子
s = GetStrelList();
% 串联去噪
e = ErodeList(Ig,s);

% 计算权重
f = GetRateList(Ig,e);

% 并联
Igo = GetRemoveResult(f,e);


% 结果
figure;
subplot(1, 2, 1); imshow(I, []); title('原图像');
subplot(1, 2, 2); imshow(Ig, []); title('噪声图像');
% imshow(I,[]) 显示灰度图像 I，根据 I 中的像素值范围对显示进行转换。imshow 使用 [min(I(:)) max(I(:))] 作为显示范围。imshow 将 I 中的最小值显示为黑色，将最大值显示为白色。有关详细信息，请参阅 DisplayRange 参数。

figure;
subplot(2, 2, 1); imshow(e.eroded_co12, []); title('串联1处理结果');
subplot(2, 2, 2); imshow(e.eroded_co22, []); title('串联2处理结果');
subplot(2, 2, 3); imshow(e.eroded_co32, []); title('串联3处理结果');
subplot(2, 2, 4); imshow(e.eroded_co42, []); title('串联4处理结果');

figure;
subplot(1, 2, 1); imshow(Ig, []); title('噪声图像');
subplot(1, 2, 2); imshow(Igo, []); title('并联去噪图像');


psnr1 = PSNR(I, e.eroded_co12);
psnr2 = PSNR(I, e.eroded_co22);
psnr3 = PSNR(I, e.eroded_co32);
psnr4 = PSNR(I, e.eroded_co42);
psnr5 = PSNR(I, Igo);
psnr_list = [psnr1 psnr2 psnr3 psnr4 psnr5];
figure; 
plot(1:5, psnr_list, 'r+-');
axis([0 6 18 24]);
set(gca, 'XTick', 0:6, 'XTickLabel', {'', '串联1', '串联2', '串联3', ...
    '串联4', '并联', ''});
grid on;
title('PSNR曲线比较');