GPU programming starts with CUDA


## Conception
CUDA（Compute Unified Device Architecture）是NVIDIA开发的并行计算平台和编程模型，用于在GPU上进行通用计算。常见概念：

1. SM (Streaming Multiprocessor):
   - SM是GPU的核心计算单元。
   - 每个SM包含多个CUDA核心、共享内存、寄存器文件等。
   - SM能够同时执行多个线程块。

2. 线程 (Thread):
   - 线程是CUDA中最基本的执行单元。
   - 每个线程执行相同的kernel函数，但处理不同的数据。
   - 线程有自己的程序计数器和寄存器状态，可以执行分支和循环。

3. 线程块 (Thread Block):
   - 线程块是线程的集合，通常是32的倍数（因为warp大小为32）。
   - 同一个线程块中的线程可以同步和共享内存。
   - 线程块被调度到单个SM上执行。
   - 线程块内的线程可以是1维、2维或3维组织的。

4. Grid (网格):
   - Grid是线程块的集合。
   - 一个kernel启动时定义了一个grid。
   - Grid可以是1维、2维或3维的。
   - Grid中的所有线程块执行相同的kernel函数。

5. Warp:
   - Warp是SM的调度单位，通常包含32个线程。
   - 同一个warp中的线程以SIMT（单指令多线程）方式执行。
   - 当warp中的线程执行路径分叉时，会发生分支分化，影响性能。

这些概念的层次关系：

> Grid > Thread Block > Thread

一些重要的关系和特点：

1. 一个Grid包含多个Thread Block。
2. 一个Thread Block包含多个Thread。
3. Thread Block被分配到SM上执行。
4. 同一个Block内的线程可以通过共享内存通信和同步。
5. 不同Block之间的线程不能直接通信。
6. kernel函数定义了单个线程的行为，CUDA运行时根据Grid和Block配置来并行执行多个线程。

在编程时，需要定义Grid和Block的维度，例如：

```cuda
dim3 blockSize(256);
dim3 threadGrid((N + blockSize.x - 1) / blockSize.x);
myKernel<<<blockSize, threadGrid>>>(args);
```

这里定义了一个一维的Block（包含256个线程）和一维的Thread（大小取决于总问题规模N）。


## Summary

1. 如何通过cmake给cuda代码添加编译选项:
```c++
target_compile_options(helloworld PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g, -0
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
```

2. 如果需要调试cuda代码， 需要添加`-G`与`-g, -0`编译选项
3. 编译工具使用`Nsight`, windows 需要利用其提供的vs studio的工具， linux可借助vscode插件

## Resource

https://github.com/NVIDIA/cuda-samples
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html