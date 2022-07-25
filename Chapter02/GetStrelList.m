function s = GetStrelList()
% 返回指定的线型算子。通过结构体成员的方式整合不同长度，角度的线型算子。
% s 算子结构体

s.co11 = strel('line',5,-45);
s.co12 = strel('line',7,-45);

s.co21 = strel('line',5,45);
s.co22 = strel('line',7,45);

s.co31 = strel('line',3,90);
s.co32 = strel('line',5,90);

s.co41 = strel('line',3,0);
s.co42 = strel('line',5,0);

% SE = strel('line',len,deg) 创建一个关于邻域中心对称的线性结构元素，长度约为 len，角度约为 deg。

% strel 对象表示一个平面形态学结构元素，该元素是形态学膨胀和腐蚀运算的重要部分。
% 平面结构元素是一个二维或多维的二值邻域，其中 true 像素包括在形态学运算中，false 像素不包括在内。
% 结构元素的中心像素称为原点，用于标识图像中正在处理的像素。
% 使用 strel 函数（如下所述）创建一个平面结构元素。您可以将平面结构元素用于二值图像和灰度图像。