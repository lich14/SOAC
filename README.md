## Dependency
numpy == 1.17.2

torch == 1.3.1

gym == 0.15.4

mujoco-py == 2.0.2.9

## Quick Start


```
python3 runSOAC.py --env Hopper-v2 --seed 0 
```

## Task

Hopper-v2, Walker2d-v2, HalfCheetah-v2, Ant-v2

## 实验思路

做了实验之后发现实验效果不错，但是学出来的option之间的相似度极高，可能是policy网络重合度太高导致的

1.针对policy网络没有重合的情况进行测试
