path = './images/sweden_input.jpg';
figure;

%% 原图
InputImage = imread(path);

subplot(3,4,1); imshow(InputImage); title('原图');
ImageInputGray = rgb2gray(InputImage);
subplot(3,4,5); imshow(ImageInputGray); title('灰度图');
subplot(3,4,9); imhist(ImageInputGray,64); title('灰度直方图')

%% 全局直方图算法
GlobalImage = RemoveFogByGlobalHisteq(InputImage);

subplot(3,4,2); imshow(GlobalImage); title('全局直方图处理图');
GlobalImageGray = rgb2gray(GlobalImage);
subplot(3,4,6); imshow(GlobalImageGray); title('灰度图');
subplot(3,4,10); imhist(GlobalImageGray,64); title('灰度直方图')

%% 局部直方图算法
LocalImage = RemoveFogByLocalHisteq(InputImage);

subplot(3,4,3); imshow(LocalImage); title('局部直方图处理图');
LocalImageGray = rgb2gray(LocalImage);
subplot(3,4,7); imshow(LocalImageGray); title('灰度图');
subplot(3,4,11); imhist(LocalImageGray,64); title('灰度直方图')

%% Retinex增强算法
RetinexImage = RemoveFogByRetinex(InputImage);

subplot(3,4,4); imshow(RetinexImage); title('Retinex增强处理图');
RetinexImageGray = rgb2gray(RetinexImage);
subplot(3,4,8); imshow(RetinexImageGray); title('灰度图');
subplot(3,4,12); imhist(RetinexImageGray,64); title('灰度直方图')