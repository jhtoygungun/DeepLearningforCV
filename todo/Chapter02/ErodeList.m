function e = ErodeList(Ig, s)
% 图像串联去噪。
% 结构体 e

e.eroded_co11 = imerode(Ig,s.co11);
e.eroded_co12 = imerode(e.eroded_co11,s.co12);

e.eroded_co21 = imerode(Ig,s.co21);
e.eroded_co22 = imerode(e.eroded_co21,s.co22);

e.eroded_co31 = imerode(Ig,s.co31);
e.eroded_co32 = imerode(e.eroded_co31,s.co32);

e.eroded_co41 = imerode(Ig,s.co41);
e.eroded_co42 = imerode(e.eroded_co41,s.co42);

% J = imerode(I,SE) 腐蚀灰度图像、二值图像或压缩二值图像 I，返回腐蚀图像 J。
% SE 是结构元素对象或结构元素对象的数组，由 strel 或 offsetstrel 函数返回。