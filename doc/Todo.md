# Todo

## 21.1.11

CMake开debug模式

```CMake
SET(CMAKE_BUILD_TYPE "Debug")
```

或者直接在shell中使用

```shell
cmake -DCMAKE_BUILD_TYPE=Debug .. #debug
cmake -DCMAKE_BUILD_TYPE=Release .. #release
```

## 21.1.13

+ 霍夫变换取点换算法，使用大根堆来代替qsort
+ 霍夫变换的新坐标系的粒度问题
+ 霍夫变换的在原图像上画线的问题
+ sobel是否确实比canny好用？是如何做出来只有一条直线的sobel的
+ 霍夫变换直线的终止处应该怎么设置
+ 自己的sobel/canny库
+ 自动化取阈值
