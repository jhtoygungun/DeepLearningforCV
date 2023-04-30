function output = RemoveFogByLocalHisteq(input)
    % RH = adapthisteq(input(:,:,1));
    % GH = adapthisteq(input(:,:,2));
    % BH = adapthisteq(input(:,:,3));

    RH = adapthisteq(input(:,:,1),'clipLimit',0.02,'Distribution','rayleigh');
    GH = adapthisteq(input(:,:,2),'clipLimit',0.02,'Distribution','rayleigh');
    BH = adapthisteq(input(:,:,3),'clipLimit',0.02,'Distribution','rayleigh');

    % J = adapthisteq(I,Name,Value) 使用名称-值对组来控制对比度增强的各个方面。

    output = cat(3,RH,GH,BH);
end
