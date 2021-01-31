# DIP-Fantasy

BIT数字图像处理大作业。试图运用大三上所学的所有新技术来完成。

## 文档

过程文档全部在`doc`文件夹下，包括项目的配置，以及`CMake`、`OpenCV`等的使用。

## 编译方法

```shell
cmake .
make
```

如果要清除编译后的缓存，使用vscode的任务clean即可。

### Debug

模块Debug：在`/lib`下有一个`/lib/debug`文件，里面装了单独编译`lib/DipFantasy.cpp`的`CMake`文件。将`DipFantasy.cpp`的main注释解除，再将vscode的调试选项设置为`module debug`就能单独调试库函数。

## 流程

滤波平滑->阈值->sobel/canny 总之要把车道线转换为单独的像素->霍夫变换->得到对应检测的线的参数
