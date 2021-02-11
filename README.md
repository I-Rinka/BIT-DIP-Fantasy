# 数字图像大作业 车道线检测 实验报告

已挂上GPL协议，抄袭，过度借鉴等**版权必纠！**

如果想查看更多与项目运行、代码相关的部分，请查看`README_old.md`

## 实现流程

在时间长达一个月的大作业实现期间，我想出了许多不同的方法来进行车道线检测，它们在大体上的实现流程相近，即：

![image-20210210211309627](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210211309627.png)

在`色彩切片`、`边界提取`处，可以选用许多不同的方法来实现。在经过一些优化和调整后，我选取了一套固定的具体实现，形成了我目前的方案，并且取得了较为不错的`0.693`的检测值：

```shell
[{"name":"Accuracy","value":0.6931994047619051,"order":"desc"},{"name":"FP","value":0.5625,"order":"asc"},{"name":"FN","value":0.4958333333333333,"order":"asc"}]
```

若经过进一步参数调优或者引入新的步骤，这个准确率有望进一步提高，后续可能的优化方案会在文章最后一部分提及。

此方法较为详细的流程如下：

![image-20210210224545440](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210224545440.png)

### 色彩切片

色彩切片处是希望通过车道线的颜色特征：白色或者黄色来提取车道线，白色和黄色单独提取后两者图像相加便得到了同时含有白色车道和黄色车道的图像。

色彩切片步骤的关键在于如何获得较为纯净的车道线，以让后续步骤更容易判断直线的位置。

#### 白色车道线切片

白色车道线选择使用球状的RGB色彩切片：如果一个点的RGB值与给定的一个颜色的RGB值的“空间坐标系距离”在一个区间内，则将其切片。

但是这有一个问题：因为不同图片的光照情况不同，因此很难选取一个作为基准的白色值——因为一个光照条件好的图像的地面的“黑色”的RGB值很有可能会相当于另一光照较差的图像的“白色”。

最初我想使用对所有图片都进行直方图均衡，再使用一个统一的白色的RGB值对他们切片。但是将这种方法应用于部分对比度较差的图片，会获得很不理想的效果：

![image-20210210215604450](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210215604450.png)

例如上面这张图，假如对它的每个通道都进行单独的直方图均衡，则每一个通道都会出现类似下图的效果：白线反而更难以辨别了。

因此我选用了另一种方法：在图像的中下部搜索RGB强度最大的像素点，以此RGB值作为这张图像的白色值，对这个白色值进行变换后再对其进行球形切片。![image-20210210220409296](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210220409296.png)

右图：对左图使用我的方法进行切片后获得的白色车道线（同时我将每个识别出来的白色车道线的RGB值都更新为了(255,255,255)）。

由于这个方法为每一张图像动态地选取了不同的白色值，因此对几乎所有的图都有较好的效果。

#### 黄色车道线切片

黄色车道线切片选择使用了在HSL色彩空间而不是RGB色彩空间下切片。

由于光照等因素的影响，不同图像的对比度差异较大，这会导致黄色的车道线的RGB值有较大的波动，会有如下图所示的情况出现。如果使用RGB切片并选取较大的切片范围的话可能有较差的结果。而在HSI色彩空间下的黄色车道线的H值（色度）却不会有较大变化，因此可以有更好的效果。

![image-20210210214039142](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210214039142.png)

通过观察，我最后选择了使用色度`30+-5`作为黄色车道线的色度进行切片。

黄色车道线切片示例：

![image-20210210221528979](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210221528979.png)

#### 切片的后处理

##### 图像相加

黄色的车道线+白色车道线和起来才是完整的、需要检测的车道线，因此我使用了图像相加，示例结果如下：

![image-20210210221752645](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210221752645.png)

##### 删除天空

部分图像的天空会有大片的白色或者较亮的蓝色，容易在白色切片时被引入。而天上的信息是我们并不需要的，因此我们需要选用一种方法排除天空对图像纯净度的印象，删掉天空以让图片尽可能的只保留有车道线。

![image-20210210222615488](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210222615488.png)

我通过直角坐标系计算与调整观察，初步选择了如上图所示的一个椭圆+长方形来作为图像保留的部分。在大部分图像上——包括上坡、平地、下坡等地平线发生变动的图像，这个范围都能较好的划分车道线与天空。图像外的天空会被删去。

天空删去前：

![image-20210210223706664](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210223706664.png)

天空删去后：

![image-20210210223656689](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210223656689.png)

##### 形态学运算

由于某些线在切片后并不是完整的，可能出现如下图所出现的“空心”情况，因此我还在对其进行了形态学运算将其进行“补全”——使用闭运算平滑直线内部黑色的细线，再使用膨胀运算填充直线。

![image-20210210224635105](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210224635105.png)

### 边界提取

边界提取这一步骤的主要任务其实是让直线

边界提取主要有两种方式（对于骨架来说，似乎“边界提取”不太准确）：

+ Sobel或Canney等梯度运算法
+ 形态学骨架

前者是对图像进行梯度运算来获得边界——一根线会出现左右两个边界，而形态学骨架（通过腐蚀、图像相加等一系列操作获得）则有更出色的效果：结果只留下中间的一根细线，因此在霍夫运算中理论上会有更出色的效果。

骨架+霍夫运算效果图：

![image-20210210225442717](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210225442717.png)

使用骨架会获得更高的准确率（`0.69` 对比 `0.70`）:

```shell
[{"name":"Accuracy","value":0.7023065476190474,"order":"desc"},{"name":"FP","value":0.5575,"order":"asc"},{"name":"FN","value":0.4891666666666667,"order":"asc"}]
```

但对比使用简单的Sobel算子进行卷积，这种方法的运行速度会慢许多，因此最后还是选择了`Sobel X`为卷积核进行边界提取。

### 霍夫变换

考虑到使用边界提取可能会出现的左右“双线”的情况，我设计了一种算法来“模糊”地得到霍夫变换后对应的直线的参数。

对粗的、是双线的直线进行霍夫变换可能出现下图所示情况。这个问题在需要选取多条车道线的本题会体现的更加严重，因此需要设法“归并”这些相似的直线。

![image-20210210233851184](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210233851184.png)

在经过变换后的霍夫坐标系空间，每一个（theta,radius）坐标都有一个值——值较大那点是原图直线的概率较大。

![image-20210210234418978](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210234418978.png)

如图上所示的霍夫空间（图片来自网络），假设我们需要选取的两根直线是A和B，但是由于车道线分布的不均匀性——即有些车道可能是完整的一个线，而有些车道线只是一些点，这会导致可能A附近的某另外一个点A2的值会大过B，这就导致了我们不能成功的检测到我们需要的两个直线。

因此我想出了一个算法来对相似的线进行归并，筛选出差异较大的直线。

![image-20210211005516029](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211005516029.png)

我使用了大致如上图所示的数据结构：一个二维链表。theta代表霍夫坐标系中直线极坐标系下的角度，radius代表霍夫坐标系中直线的极坐标系距离。

其原理如下：

+ 使用前先设定一个阈值，如果theta和radius与目前已知的点的差在阈值范围之内，则说明他们是相似的直线，则将其挂在一个节点之下，作为其的子节点，如A和A1、A2的关系。如果这两个参数中任意一个的差大于已知阈值，则创建一个新的节点，如B和A的关系。
+ 每次插入一个子节点都要实时更新`Hough_Node`的内容。
  + 将属性`hough_value`与当前的值做对比并相加，这个的大小说明了此直线出现的可能情况。值越大说明这越可能出现一根线。
  + `theta_average`和`radius_average`的值实时与`hough_value`最大的子节点保持一致。用`theta_average`和`radius_average`表示这簇霍夫点所所代表的直线。
+ 最后读取时，将`Hough_Node`放入大根堆中，以`hough_value_sum`判定大小。

![image-20210211011203180](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211011203180.png)

以上图为例说明算法：假设新进入一个`hough_value`大于给定阈值的霍夫节点B2，并且设定`theta`和`raidus`的阈值范围为5，该节点发生了如下流程：

+ 这个节点的`theta`和`raidus`与节点A的`theta_average`和`radius_average`相对比，其差值大于5，因此被A拒绝，继续遍历到下一节点B
+ 与节点B的`theta_average`和`radius_average`小于阈值，因此被B接收，作为B的子节点被插入
+ 由于节点B2的`hough_value`非常大，因此此时应该更新节点B的`theta_average`和`radius_average`，以节点B2的`theta`和`raidus`代表节点B附近所有的直线的参数。同时更新`hough_value_sum`，以表示这一簇节点的直线的霍夫值。

通过这样的一种相似算法，我可以有效的获得应该获得的直线。

当取直线时，霍夫值大的节点被优先取出，由于这个节点的`theta_average`和`radius_average`代表了这周围的所有直线，同时这个值还是来自这附近霍夫值最大的一点，因此对应原图的准确的一条单直线会被取出。

结果示例：![image-20210211012854501](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211012854501.png)

面对一些有较多点的密集的粗线，该方法也能准确的识别出这部分区域对应的单线，而不是输出许多条线。

### 参数选取

经过我的算法的霍夫变换后，在大多数情况下输出的直线都是能满足要求的，但依然需要设计一套法则来判断输出的直线是否满足车道线的要求。

通过观察图像，我选取了几个比较简单的法则：

+ 车道线在图像上方的截距不能超出图像太远。我选取了-0.3\*图像宽度到1.3\*图像宽度这个距离。
+ 车道线在图像上分布应该是“八”字形

因为相近距离的直线的问题已经在自己设计的霍夫变换处理算法中解决，故这里便不设计通过直线之间的距离进行筛选的选定法则。



## 代码详解

### 项目主要运行文件

本次大作业中，和代码运行关系紧密的文件如下：

```shell
│
├── judge
│   ├── groundtruth.json
│   ├── lane.py
│   └── predict.json
├── lib
│   ├── DipFantasy.cpp
│   └── DipFantasy.h
├── get_predict_json.py
└── Main.cpp
```

- `/judge`目录下放置了评测机`lane.py`和跑分答案`groundtruth.json`。项目会在此目录输出预测值`predict.json`，并运行评测机`lane.py`对照`groundtruth.json`来进行跑分。

- `/lib`目录即为`DipFantasy.h`和其对应的源文件`DipFantacy.cpp`的代码，也就是本项目所试图构建的图像库。这个图像库存放了我自定义的图像数据类型以及基本的图像操作，这样就能实现不使用Open CV也能方便的处理图象。根目录的`Main.cpp`通过调用这个库完成图像处理流程。
- `get_predict_json.py`为生成`lib/predict.json`的python脚本，它会调用C++编写的可执行文件并向其输入图片，最后将检测到的车道线数据转换为json。
- `Main.cpp`为主程序的入口，通过调用`lib`中定义的数据类型以及方法来实现处理流程。

### 自定义图象处理库——DipFantacy

#### 类与接口

为了方便进行图像处理，这个库中有四个重要的对象`DF_Mat`、`DF_IMG`、`DF_Color_IMG`、`DF_Kernel`，它们之间的继承关系如下：

![image-20210211090152527](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211090152527.png)

`DF_Mat`定义了一些最接近底层的最接近“矩阵”而不是图像的基本操作——如获得矩阵的行列值等。最主要的目的是规定接口以方便`DF_IMG`和`DF_Kernel`等相互调用。

---

`DF_IMG`继承自`DF_Mat`，代表了图象。它可接收Open CV的`cv::Mat`类型的图像，并依此示例化此对象。这个类提供了许多和图象处理有关的方法，如：

```C++
void DoConvolution(DF_Kernel kernel);//对图像进行卷积
void DoErosion(DF_Kernel kernel);//形态学腐蚀
void DoDilation(DF_Kernel kernel);//形态学膨胀
void DoMultiply(DF_IMG &mask);//图像相乘
void DoPlus(DF_IMG &other);//图像相加
void DoThreshold(DF_TYPE_INT Threshold);//阈值
void DoHistEqualization();//直方图均衡
DF_TYPE_INT *GetPoint(int cols, int rows);//获得对应坐标的像素点
```

其中最重要的应是`GetPoint`方法，它可以被视作对`cv::Mat`的像素点的一种映射，通过调用`GetPoint`，类内的其他方法或是在类外使用类都可以很轻易的对图像矩阵中的像素点进行修改。

在图像卷积等`Do`系列操作中，和Open CV的风格不同的是，这个库是直接在原图上进行操作的——这样省去了open CV那样在input和output间传来传去的问题。但也因此使得如果不希望原图破坏，需要使用`=`赋值运算符来备份原图。

---

`DF_Color_IMG`继承自`DF_IMG`，为彩色图像进一步地提供了方便的处理函数——可以单独的处理一个通道了。（其实`DF_IMG`也能直接处理彩色图像，但是通常会直接同时处理三个通道）

对`DF_IMG`进行了如下的扩充：

```c++
DF_TYPE_INT *GetPoint(int row, int col, RGB channel);
void DoColorSlicing(DF_TYPE_INT RGB_Value[3], int radius);
```

其中`RGB`是自定义的枚举类型，通过这个可以更自然的方式单独处理某个通道，如：

```c++
 *this->GetPoint(i, j, R) = 0
```

代表将红色R通道的(i,j)点置0。

---

`DF_Kernel`继承自`DF_Mat`，代表卷积核。因为卷积核与普通图像不同，因此在`DF_Mat`处便与图像对象分道扬镳。一个卷积核有典型的如下特征：

+ 卷积核中的点通常不是整形，有很多情况下是浮点型
+ 卷积核通常很小

通过枚举类型，我预定义了许多内核以方便创建。其中的一些如所示：

```c++
enum PREDEFINED_KERNEL
    {
        GaussianKernel,
        BoxKernel,
        SobelKernelX,
        SobelKernelY,
    ...
    };
```

有了这样的一种数据结构，用户就能以一种相对优雅的方式进行卷积。比如对`input`图像进行大小为5的高斯滤波：

```c++
input.DoConvolution(DF_Kernel(GaussianKernel, 5));
```

---

其他的和图像对象不是那么紧密的一些如霍夫变换以及另外的一些功能便放在了类的外面。

#### 详细实现

对于本次大作业而言最重要的方法是：

+ 卷积
+ 色彩切片
+ 图像相加

因此以这三个方法为例说明这个图象库是怎么工作的

##### 卷积

```c++
void DF_IMG::DoConvolution(DF_Kernel kernel)
{
    int row = this->row_size;
    int col = this->col_size;
    int kernel_row = kernel.GetRowSize();
    int kernel_col = kernel.GetColSize();
    int l = (kernel_row / 2);
    int u = (kernel_col / 2);
    Mat temp = OCV_Mat.clone();
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            DF_TYPE_INT *now_point = OCV_Util::GetPoint<DF_TYPE_INT>(OCV_Mat, i, j);
            if (now_point != NULL)
            {
                for (int c = 0; c < OCV_Mat.channels(); c++)
                {
                    *(now_point + c) = 0;
                    double ans = 0;
                    for (int i2 = -l; i2 < l + 1; i2++)
                    {
                        for (int j2 = -u; j2 < u + 1; j2++)
                        {
                            DF_TYPE_INT *p = (OCV_Util::GetPoint<DF_TYPE_INT>(temp, i + i2, j + j2));
                            if (p != NULL)
                            {
                                // double t = *p;
                                ans += (DF_TYPE_FLOAT) * (p + c) * ((DF_TYPE_FLOAT)*kernel.GetPoint(i2 + l, j2 + u));
                            }
                        }
                    }
                    if (ans >= 255.0 || ans <= -256.0)
                    {
                        ans = 255.0;
                    }
                    if (ans <= 0)
                    {
                        ans = -ans;
                    }
                    *(now_point + c) = (DF_TYPE_INT)(ans);
                }
            }
        }
    }
}
```

因为对图像卷积会破坏此图像原有的内容，因此在开始前先将这个图像的内容复制到了另一个open CV的`cv::Mat`中。如果有必要，未来也可以直接用自己设计的图像类型`DF_IMG`代替。

卷积时会进行边界检查，如果`p==NULL`则不进行卷积。同时从`for (int c = 0; c < OCV_Mat.channels(); c++)`也可看出，对`DF_IMG`来说，卷积是同时对三个通道进行的。

在最后的一个步骤里，由于卷积所得结果的`ans`的类型是`double`，而图像像素`DF_TYPE_INT`的实质是无符号短整形`short`，因此需要对其进行值检测，以防运算溢出。

---

##### 色彩切片

一个色彩切片的典型用法为：

```c++
DF_TYPE_INT rgb_w[3] = {0xF0, 0xF0, 0xF0};
int color_radius = 70;
w_mask.DoColorSlicing(rgb_w, color_radius);
```

首先我们定义了白色的rgb值`rgb_w`，接着规定了色彩切片的半径，最后使用`w_mask.DoColorSlicing(rgb_w, color_radius);`就能在原图`w_mask`上进行色彩切片。

如果需要对HSI切片，则需要使用函数`int Get_HSI_H(DF_TYPE_INT R, DF_TYPE_INT G, DF_TYPE_INT B)`

色彩切片的具体代码如下：

```c++
void DF_Color_IMG::DoColorSlicing(DF_TYPE_INT RGB_Value[3], int radius)
{
    for (int i = 0; i < this->row_size; i++)
    {
        for (int j = 0; j < this->col_size; j++)
        {
            int l_r = (int)*this->GetPoint(i, j, R) - (int)RGB_Value[0];
            int l_g = (int)*this->GetPoint(i, j, G) - (int)RGB_Value[1];
            int l_b = (int)*this->GetPoint(i, j, B) - (int)RGB_Value[2];
            if ((l_r * l_r + l_b * l_b + l_g * l_g > radius * radius))
            {
                *this->GetPoint(i, j, R) = 0;
                *this->GetPoint(i, j, G) = 0;
                *this->GetPoint(i, j, B) = 0;
            }
        }
    }
}
```

使用立体空间坐标系的公式算出距离判断条件后，便能决定是否将原图某像素点的色彩值留下。如果色彩值在范围之外，则置为0。

---

##### 图像相加

图像相加十分简单，总而言之是遍历图像，再将对应像素的RGB值相加。

```c++
void DF_Color_IMG::DoPlus(DF_Color_IMG &other)
{
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row_size; i++)
        {
            for (int j = 0; j < col_size; j++)
            {
                *this->GetPoint(i, j, (RGB)c) += *other.GetPoint(i, j, (RGB)c);
            }
        }
    }
}
```

### Main.cpp

在`Main.cpp`中，在图像处理的流程外，又另外定义了如下函数，如：

+ 得到黄色线`GetYellowLine`
+ 删除天空`DeleteSky`
+ 得到图像白点值`GetWhite`

```c++
//通过HSI模型得到满足要求的黄色线
DF_Color_IMG GetYellowLine(DF_Color_IMG &input)
{
    DF_Color_IMG output = input;
    for (int i = 0; i < output.GetRowSize(); i++)
    {
        for (int j = 0; j < output.GetColSize(); j++)
        {
            DF_TYPE_INT *R = output.GetPoint(i, j, DF_Color_IMG::R);
            DF_TYPE_INT *G = output.GetPoint(i, j, DF_Color_IMG::G);
            DF_TYPE_INT *B = output.GetPoint(i, j, DF_Color_IMG::B);
            int H = Get_HSI_H(*R, *G, *B);
            DF_TYPE_FLOAT S = Get_HSI_S(*R, *G, *B);
            int range = 5;
            if (S >= 0.40 && ((H >= 30 - range && H <= 30 + range) || (H >= 40 - range && H <= 40 + range)))
            {
            }
            else
            {
                *R = 0;
                *G = 0;
                *B = 0;
            }
        }
    }
    return output;
}
```

```c++
//将前文所说的椭圆的天空部分删除
void DeleteSky(DF_IMG &input)
{
    for (int i = 0; i < input.GetRowSize(); i++)
    {
        for (int j = 0; j < input.GetColSize(); j++)
        {
            int x = i - 480;
            int y = j - 640;
            if (i > 360 || (double)x * x / (200.0 * 200.0) + (double)(y * y) / (800.0 * 800.0) <= 1)
            {
            }
            else
            {
                *input.GetPoint(i, j) = 0;
            }
        }
    }
}
```

```c++
//在车道附近遍历，得到此图像的白色值
DF_TYPE_INT GetWhite(DF_Color_IMG &img)
{
    DF_TYPE_INT max = 0;
    for (int i = 400; i < img.GetRowSize(); i++)
    {
        for (int j = img.GetColSize() * 0.2; j < img.GetColSize() * 0.8; j++)
        {
            DF_TYPE_INT t_min = 255;
            for (int c = 0; c < 3; c++)
            {
                if (*img.GetPoint(i, j, (DF_Color_IMG::RGB)c) < t_min)
                {
                    t_min = *img.GetPoint(i, j, (DF_Color_IMG::RGB)c);
                }
            }
            if (max < t_min)
            {
                max = t_min;
            }
        }
    }
    return max;
}
```

#### main函数

```c++
int main(int argc, char const *argv[])
{

    if (argc != 2)
    {
        return -1;
    }
    Mat image = imread(argv[1]);

    DF_Color_IMG input(image);

    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

    DF_Color_IMG y_mask = GetYellowLine(input);

    //得到白色线
    DF_Color_IMG w_mask = input;
    DF_TYPE_INT white_max = GetWhite(input);
    white_max *= 0.9;
    DF_TYPE_INT rgb_w[3] = {white_max, white_max, white_max};
    int color_radius = 70;
    w_mask.DoColorSlicing(rgb_w, color_radius);

    w_mask.DoPlus(y_mask);

    DF_IMG mask = w_mask.ToGrey();

    //删除天空
    DeleteSky(mask);
    
    //形态学运算补齐直线
    mask.DoDilation(DF_Kernel(BoxKernel, 7));
    mask.DoErosion(DF_Kernel(BoxKernel, 7));
    mask.DoDilation(DF_Kernel(BoxKernel, 5));

    DF_IMG grey = mask;

    //边界提取
    grey.DoConvolution(DF_Kernel(SobelKernelX, 3));
    //霍夫变换
    HoughTransition HT(grey, 10);

    int count = 0;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();

        //满足输出条件的线
        if (LineRule(grey, now.theta_average, now.radius_average))
        {
            //输出每个满足条件的线的极坐标
            // DrawLineToImage(grey, now.radius_average, now.theta_average);
            if (count >= 4)
            {
                break;
            }
            cout << now.theta_average << " " << now.radius_average << endl;
            DrawLineToImage(mask, now.radius_average, now.theta_average);
            count++;
        }

        HT.node_queue.pop();
    }
}
```

main函数的具体步骤与前文的流程图一致，即：

![image-20210210224545440](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210210224545440.png)

在使用Open CV的`Mat image = imread(argv[1]);`读入图像后并调用` DF_Color_IMG input(image);`生成自己的数据格式后，图象处理便与Open CV解耦了。

## 总结与思考

### 可行的改进

本方法最高可在跑分中取得`0.7`以上的准确率，但事实上若经过参数调优或引入新的步骤可有望达到更高的准确率。

本方法的关键是如何在色彩切片时得到纯净的车道线图像。因此，虽然本方法在大多数“正常”的图像中可以实现几乎100%的准确率，但是有如下特征的图象则无法得到较好效果：

+ 地面附近有白色色块
+ 车道附近有黄色沙滩

#### 白色切片异常

有些图片附近由于颜色的相似性，它们的白色可能会被误判为车道线。

![image-20210211103812752](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211103812752.png)

![image-20210211103851159](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211103851159.png)

#### 黄色切片异常

黄色沙滩产生影响的原因：有着和车道线相似的H值，处在车道线的`30+-5`区间中

![image-20210211103425271](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211103425271.png)

![image-20210211103643952](https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211103643952.png)

#### 异常图片分析

异常图片案例：

黄色影响：

<img src="https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211104203372.png" alt="image-20210211104203372" style="zoom:50%;" />

白色影响：

<img src="https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211104220338.png" alt="image-20210211104220338" style="zoom:50%;" />

如果直接对这些图像按流程进行霍夫变换，则会出现以下效果：

<img src="https://cdn.jsdelivr.net/gh/I-Rinka/picTure//image-20210211104128899.png" alt="image-20210211104128899" style="zoom:50%;" />

但通过图像观察，可以得出这些图像共同的性质：

+ 产生不良影响的色块为**大面积**色块

因此理论上可以使用**傅里叶变换**对其进行低频滤波来排除这些干扰，这样应该可在一定程度上提高识别准确率。

但由于傅里叶变换手工实现起来较为复杂，因此最终没有在代码内添加。

### 总结

我实现了一个小的图象库，以简化图象处理的流程。

同时想出了一个车道线检测的方法，它可以实现较好的识别率，同时也有一定的提升空间。

