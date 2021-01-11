# OpenCV大作业

## OpenCV使用

### 创建矩阵

```C++
Mat image(1000,1000,CV_8UC3,Scalar(255,200,255));//前面两个参数指定了矩阵的长宽,CV_8UC3指定了"8字节 U无符号 3 C通道数（颜色的种类）",Scalar则指定了每个像素点的RGB值
```

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

## Git使用

### 初始化

```shell
git init # 初始化本地仓库
git add * # 跟踪
git rm --cached CMakeFiles -r # 取消跟踪某一文件夹
```

### 查看

`git status -s`

## Cmake使用

Cmake一般会单独建一个文件夹来放`release`或者`release`。

Cmake的配置文件会用给出的`cmake <CMake list dic>`里的，而cmake生成的中间文件则是在当前文件夹中。

## GDB使用
