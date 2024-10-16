GPU programming starts with CUDA


## Conception
CUDA��Compute Unified Device Architecture����NVIDIA�����Ĳ��м���ƽ̨�ͱ��ģ�ͣ�������GPU�Ͻ���ͨ�ü��㡣�������

1. SM (Streaming Multiprocessor):
   - SM��GPU�ĺ��ļ��㵥Ԫ��
   - ÿ��SM�������CUDA���ġ������ڴ桢�Ĵ����ļ��ȡ�
   - SM�ܹ�ͬʱִ�ж���߳̿顣

2. �߳� (Thread):
   - �߳���CUDA���������ִ�е�Ԫ��
   - ÿ���߳�ִ����ͬ��kernel������������ͬ�����ݡ�
   - �߳����Լ��ĳ���������ͼĴ���״̬������ִ�з�֧��ѭ����

3. �߳̿� (Thread Block):
   - �߳̿����̵߳ļ��ϣ�ͨ����32�ı�������Ϊwarp��СΪ32����
   - ͬһ���߳̿��е��߳̿���ͬ���͹����ڴ档
   - �߳̿鱻���ȵ�����SM��ִ�С�
   - �߳̿��ڵ��߳̿�����1ά��2ά��3ά��֯�ġ�

4. Grid (����):
   - Grid���߳̿�ļ��ϡ�
   - һ��kernel����ʱ������һ��grid��
   - Grid������1ά��2ά��3ά�ġ�
   - Grid�е������߳̿�ִ����ͬ��kernel������

5. Warp:
   - Warp��SM�ĵ��ȵ�λ��ͨ������32���̡߳�
   - ͬһ��warp�е��߳���SIMT����ָ����̣߳���ʽִ�С�
   - ��warp�е��߳�ִ��·���ֲ�ʱ���ᷢ����֧�ֻ���Ӱ�����ܡ�

��Щ����Ĳ�ι�ϵ��

> Grid > Thread Block > Thread

һЩ��Ҫ�Ĺ�ϵ���ص㣺

1. һ��Grid�������Thread Block��
2. һ��Thread Block�������Thread��
3. Thread Block�����䵽SM��ִ�С�
4. ͬһ��Block�ڵ��߳̿���ͨ�������ڴ�ͨ�ź�ͬ����
5. ��ͬBlock֮����̲߳���ֱ��ͨ�š�
6. kernel���������˵����̵߳���Ϊ��CUDA����ʱ����Grid��Block����������ִ�ж���̡߳�

�ڱ��ʱ����Ҫ����Grid��Block��ά�ȣ����磺

```cuda
dim3 blockSize(256);
dim3 threadGrid((N + blockSize.x - 1) / blockSize.x);
myKernel<<<blockSize, threadGrid>>>(args);
```

���ﶨ����һ��һά��Block������256���̣߳���һά��Thread����Сȡ�����������ģN����


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