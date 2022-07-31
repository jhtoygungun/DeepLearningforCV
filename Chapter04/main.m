clc; clear all; close all;
warning off all;
I = imread('images\\1.jpg');

%% 二值化
I1 = Image_Normalize(I, 1); % 图像归一化处理
hsize = [3 3];
sigma = 0.5;
I2 = Image_Smooth(I1, hsize, sigma, 1);% 图像平滑滤波
I3 = Gray_Convert(I2, 1); % 图像灰度化
bw2 = Image_Binary(I3, 1); % 灰度图像二值化
figure; subplot(1, 2, 1); imshow(I, []); title('原图像'); subplot(1, 2, 2); imshow(bw2, []); title('二值化图像');

%% 图像校正
[~, ~, xy_long] = Hough_Process(bw2, I1, 1); % Hough检测
angle = Compute_Angle(xy_long); % 
[I4, bw3] = Image_Rotate(I1, bw2, angle*1.8, 1);
[bw4, Loc1] = Morph_Process(bw3, 1);
[Len, XYn, xy_long] = Hough_Process(bw4, I4, 1);

%% 图像分割
[bw5, bw6] = Region_Segmation(XYn, bw4, I4, 1);
[stats1, stats2, Line] = Location_Label(bw5, bw6, I4, XYn, Loc1, 1);
[Dom, Aom, Answer, Bn] = Analysis(stats1, stats2, Line, I4);