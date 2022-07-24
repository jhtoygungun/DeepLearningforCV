function In = RemoveFogByRetinex(f)
    %
    % Retinex是一种常用的建立在科学实验和科学分析基础上的图像增强方法，它是Edwin.H.Land于1963年提出的。就跟Matlab是由Matrix和Laboratory合成的一样，Retinex也是由两个单词合成的一个词语，他们分别是retina 和cortex，即：视网膜和皮层。Land的retinex模式是建立在以下三个假设之上的：
    % （1）真实世界是无颜色的，我们所感知的颜色是光与物质的相互作用的结果。我们见到的水是无色的，但是水膜—肥皂膜却是显现五彩缤纷，那是薄膜表面光干涉的结果。
    % （2）每一颜色区域由给定波长的红、绿、蓝三原色构成的；
    % （3）三原色决定了每个单位区域的颜色。
    % Retinex理论的基础理论是物体的颜色是由物体对长波（红色）、中波（绿色）、短波（蓝色）光线的反射能力来决定的，而不是由反射光强度的绝对值来决定的，物体的色彩不受光照非均匀性的影响，具有一致性，即retinex是以色感一致性（颜色恒常性）为基础的。不同于传统的线性、非线性的只能增强图像某一类特征的方法，Retinex可以在动态范围压缩、边缘增强和颜色恒常三个方面打到平衡，因此可以对各种不同类型的图像进行自适应的增强。
    % 40多年来，研究人员模仿人类视觉系统发展了Retinex算法，从单尺度Retinex算法改进成多尺度加权平均的Retinex算法，再发展成彩色恢复多尺度Retinex算法。

    %提取图像的R、G、B分量
    fr = f(:, :, 1);
    fg = f(:, :, 2);
    fb = f(:, :, 3);

    %数据类型归一化
    mr = mat2gray(im2double(fr));
    mg = mat2gray(im2double(fg));
    mb = mat2gray(im2double(fb));


    %% 定义alpha参数
    alpha = randi([80 100], 1)*20;
    %定义模板大小
    n = floor(min([size(f, 1) size(f, 2)])*0.5);
    %计算中心
    n1 = floor((n+1)/2);
    for i = 1:n
        for j = 1:n
            %高斯函数
            b(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*alpha))/(pi*alpha);
        end
    end
    %卷积滤波
    nr1 = imfilter(mr,b,'conv', 'replicate');
    ng1 = imfilter(mg,b,'conv', 'replicate');
    nb1 = imfilter(mb,b,'conv', 'replicate');
    ur1 = log(nr1);
    ug1 = log(ng1);
    ub1 = log(nb1);
    tr1 = log(mr);
    tg1 = log(mg);
    tb1 = log(mb);
    yr1 = (tr1-ur1)/3;
    yg1 = (tg1-ug1)/3;
    yb1 = (tb1-ub1)/3;


    %% 定义beta参数
    beta = randi([80 100], 1)*1;
    %定义模板大小
    x = 32;
    for i = 1:n
        for j = 1:n
            %高斯函数
            a(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*beta))/(6*pi*beta);
        end
    end
    %卷积滤波
    nr2 = imfilter(mr,a,'conv', 'replicate');
    ng2 = imfilter(mg,a,'conv', 'replicate');
    nb2 = imfilter(mb,a,'conv', 'replicate');
    ur2 = log(nr2);
    ug2 = log(ng2);
    ub2 = log(nb2);
    tr2 = log(mr);
    tg2 = log(mg);
    tb2 = log(mb);
    yr2 = (tr2-ur2)/3;
    yg2 = (tg2-ug2)/3;
    yb2 = (tb2-ub2)/3;


    %% 定义eta参数
    eta = randi([80 100], 1)*200;
    for i = 1:n
        for j = 1:n
            %高斯函数
            e(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*eta))/(4*pi*eta);
        end
    end
    %卷积滤波
    nr3 = imfilter(mr,e,'conv', 'replicate');
    ng3 = imfilter(mg,e,'conv', 'replicate');
    nb3 = imfilter(mb,e,'conv', 'replicate');
    ur3 = log(nr3);
    ug3 = log(ng3);
    ub3 = log(nb3);
    tr3 = log(mr);
    tg3 = log(mg);
    tb3 = log(mb);
    yr3 = (tr3-ur3)/3;
    yg3 = (tg3-ug3)/3;
    yb3 = (tb3-ub3)/3;
    dr = yr1+yr2+yr3;
    dg = yg1+yg2+yg3;
    db = yb1+yb2+yb3;
    cr = im2uint8(dr);
    cg = im2uint8(dg);
    cb = im2uint8(db);

    % 集成处理后的分量得到结果图像
    In = cat(3, cr, cg, cb);
end