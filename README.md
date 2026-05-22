# 📍 Dataset-Distillation-FL-FedUSD
Official PyTorch implementation of paper (ICML 2026) 🤩
"FedUSD：Unbiased Synthetic Data for Federated Learning" 
>[Weiying Xie](https://scholar.google.com/citations?user=y0ha5lMAAAAJ&hl=zh-CN), [Chenhe Hao](https://github.com/HCH-XDU), [Haozhi Shi](https://github.com/xirihao), [Jitao Ma](https://github.com/majitao-xd), [Daixun Li](https://scholar.google.cz/citations?user=gaiP4-IAAAAJ&hl=zh-CN&oi=ao), [Jiazhe Li](https://github.com/JZheL), [Hengyi Wang](https://github.com/hywhyw111?tab=repositories), [Leyuan Fang](https://scholar.google.cz/citations?user=Gfa4nasAAAAJ&hl=zh-CN&oi=ao), [Yunsong Li](https://ieeexplore.ieee.org/author/37292407800)<br>
>XDU and HNU
![image](https://github.com/HCH-XDU/Dataset-Distillation-FL-FedUSD/blob/main/framework.png)
## 📝 Abstract
Aggregation-Free Federated Learning enables joint training by sharing synthetic data, aiming to eliminate data heterogeneity across clients. However, existing methods fail to explicitly separate the principal and residual components of dataset, leading to biased synthetic data. In this paper, we propose a novel Unbiased Synthetic Data optimization method FedUSD for Aggregation-Free Federated Learning, which is achieved by exploring the High-energy Orthogonal Base (HOB) and variance of dataset in feature space. Our FedUSD is inspired by the discovery that principal component concentrates in HOB while residual component independently reflects in variance, regardless of networks. Based on the observation, we develop a method that mathematically optimizes synthetic data by matching both HOB and variance with those of real data. Besides, we experimentally show the superior effectiveness of leveraging HOB and variance to separately extract the principal and residual components over existing methods. We also theoretically prove that FedUSD achieves unbiased synthetic data and thus convergence. Without introducing any constraints, FedUSD thereby yields significant improvements over the state-of-the-arts in terms of global model performance, under equivalent communicational costs. For example, on the SVHN dataset, FedUSD improves 6.74\% to 30.82\% which is higher than others with Dirichlet coefficient $\alpha=0.01$.

## 👩🏻‍💻 Usage and Examples
Use following steps you can reproduce FedUSD on CIFAR10, CIFAR100, SVHN, TINY-IMAGENET. Here we use CIFAR10 as an example, the detailed training setting can be found in our paper.
### Environment 🌏
We conduct our experiments on 3090 GPUs in an environment configured as follows:
torch >= 1.8
torchvision >= 0.9
numpy >= 1.19

Readers do not need to replicate our setup exactly.
  
### Run 🏃🏻‍♀️
* `python main_fedusd.py`

### 🙋🏻‍♀️ Citation

If you find our codes useful for your research, please cite our paper. 🤭

```
@inproceedings{Hao2025FedCS,
  title={FedUSD：Unbiased Synthetic Data for Federated Learning},
  author={Weiying, Xie and Chenhe, Hao and Haozhi, Shi and Jitao, Ma and Daixun, Li and Jiazhe, Li and Hengyi, Wang and Leyuan, Fang and Yunsong, Li},
  booktitle={Proceedings of the Forty-third International Conference on Machine Learning (ICML)},
  year={2026},
}
```

