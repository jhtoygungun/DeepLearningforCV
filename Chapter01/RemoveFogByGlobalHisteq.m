function output = RemoveFogByGlobalHisteq(input)
    % RGB 分量的提取
    R = input(:,:,1);
    G = input(:,:,2);
    B = input(:,:,3);

    RH = histeq(R);
    GH = histeq(G);
    BH = histeq(B);
    % J = histeq(I) 变换灰度图像 I，以使输出灰度图像 J 的直方图具有 64 个 bin 且大致平坦。

    output = cat(3,RH,GH,BH); 
    %C = cat(dim,A1,A2,…,An) 沿维度 dim 串联 A1、A2、…、An。
end