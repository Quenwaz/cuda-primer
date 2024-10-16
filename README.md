GPU programming starts with CUDA

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