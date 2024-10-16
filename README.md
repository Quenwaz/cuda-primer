GPU programming starts with CUDA

## Summary

1. ���ͨ��cmake��cuda������ӱ���ѡ��:
```c++
target_compile_options(helloworld PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g, -0
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
```

2. �����Ҫ����cuda���룬 ��Ҫ���`-G`��`-g, -0`����ѡ��
3. ���빤��ʹ��`Nsight`, windows ��Ҫ�������ṩ��vs studio�Ĺ��ߣ� linux�ɽ���vscode���

## Resource

https://github.com/NVIDIA/cuda-samples
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html