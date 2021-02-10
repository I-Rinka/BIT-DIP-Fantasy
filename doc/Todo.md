# Todo

## 两个目标

+ 通过色彩分割切出车道线
+ 将分割出的粗线想办法变细

其余影响准确率的因素就是霍夫变换直线的筛选了

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

## 21.2.9

+ 如何取出阴天的直线
+ 中值滤波

## 黄色的HSI

```shell
hsl(33, 19%, 45%)
hsl(32, 38%, 47%)
hsl(33, 85%, 64%)
hsl(30, 29%, 59%)
hsl(28, 44%, 60%)
hsl(31, 83%, 67%)
hsl(44, 81%, 69%)
hsl(32, 99%, 70%)
hsl(36, 96%, 71%)
hsl(29, 58%, 67%)
```

因此黄色直线的色调主要围绕在30+-4，对于某些天气下，也有可能是44
