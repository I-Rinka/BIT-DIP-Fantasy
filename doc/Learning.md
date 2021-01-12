# 数图大作业

## OpenCV使用

### 官方教程

Linux版本的安装:<https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html>

第一个OpenCV程序:<https://docs.opencv.org/master/db/df5/tutorial_linux_gcc_cmake.html>

OpenCV图像储存的基本数据结构Mat:<https://docs.opencv.org/master/d6/d6d/tutorial_mat_the_basic_image_container.html>

OpenCV矩阵的遍历:<https://docs.opencv.org/master/db/da5/tutorial_how_to_scan_images.html>

### 创建矩阵

```C++
Mat image(1000,1000,CV_8UC3,Scalar(255,200,255));//前面两个参数指定了矩阵的长宽,CV_8UC3指定了"8字节 U无符号 3 C通道数（颜色的种类）",Scalar则指定了每个像素点的RGB值
```

---

Matlab风格的简单矩阵创建:

```C++
    Mat E = Mat::eye(4, 4, CV_64F);
    cout << "E = " << endl << " " << E << endl << endl;
    Mat O = Mat::ones(2, 2, CV_32F);
    cout << "O = " << endl << " " << O << endl << endl;
    Mat Z = Mat::zeros(3,3, CV_8UC1);
    cout << "Z = " << endl << " " << Z << endl << endl;
```

---

简易的、自己指定每个像素的值和矩阵形状的矩阵的创建:

```C++
Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
```

矩阵随机化(先创建一个空矩阵，再对矩阵内元素随机化):

```C++
    Mat R = Mat(3, 2, CV_8UC3);
    randu(R, Scalar::all(0), Scalar::all(255));
```

### 矩阵遍历

```C++
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols * channels;
    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}
```

### 读入图像

读入灰度图:

```C++
    Mat img = imread(filename, IMREAD_GRAYSCALE);
```

## Git使用

### 初始化

```shell
git init # 初始化本地仓库
git add * # 跟踪
git rm --cached CMakeFiles -r # 取消跟踪某一文件夹
```

### 查看

`git status -s`

## CMake使用

CMake是一个跨平台的C/C++的编译安装工具。CMake其实就相当于是一个脚本文件，通过`CMakeList.txt`中的参数，可以很轻松的生成复杂的`Makefile`，而无需自己编写。还有一个原因是，在跨平台环境下不同操作系统的工具链是不一样的，比如Windows中就不是使用make。而通过CMake可以生成跨平台的编译文件。

CMake一般会单独建一个文件夹来放`release`或者`debug`。

CMake的配置文件会用给出的`cmake <CMake list dic>`里的，而cmake生成的中间文件则是在当前文件夹中。

本项目使用根目录来存放`CMakeLists.txt`，同时也直接在根目录存放CMake的各种编译文件，如果需要清除编译安装环境则使用vscode配置的`tasks`中的`clean`任务以删除文件夹。

### CMake多文件编译

如果要编译多个源文件，那么有两种方式:

+ 单独开一个lib目录，将自己的源文件单独编译成一个二进制库，再使用主函数进行链接
+ 将其他的源文件添加到生成可执行文件的列表，与主函数同时编译

这里使用第二种方法

以21.1.12本项目的CMakeList为例:

```CMake
cmake_minimum_required(VERSION 2.8)
project( DipFantasy )

SET(CMAKE_BUILD_TYPE "Debug")#设置debug

find_package( OpenCV REQUIRED )#找到OpenCV的库

include_directories( ./lib )#自己写的头文件
aux_source_directory( ./lib My_LIBS )#实现头文件的各种源文件


include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DipFantasy Main.cpp ${My_LIBS} )

target_link_libraries( DipFantasy ${OpenCV_LIBS} )#链接OpenCV的库

```

这个CMake文件的基本流程是: 设置项目的属性(版本号、debug等)->找到OpenCV的库->添加自己库的include文件和源文件(所有源文件最后会储存到列表`My_LIBS`中) ->添加OpenCV的头文件->添加生成文件->把生成文件和OpenCV库链接

通过`aux_source_directory( ./lib My_LIBS )`可以把所有源文件的地址都存入`${My_LIBS}`中，这样就可以直接调用`add_executable( DisplayImage DisplayImage.cpp ${My_LIBS} )`同时编译主函数文件`Main.cpp`和`.\lib`文件夹下的所有源文件了。

## GDB使用
