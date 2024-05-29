# <div align="center">[KDD-24] ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation </div>

[![KDD](https://img.shields.io/badge/KDD-2024-blue.svg?style=flat-square)](#)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=ImputeFormer&color=red&logo=arxiv)](https://arxiv.org/abs/2312.01728)

**[Updata 20240526]: We will release the full model training and evaluation codes soon after carefully preparing the final version.**

<div align=center>
	<img src="./Imputeformer_introduction.png" alt="Example of the sparse spatiotemporal attention layer."/>
	<p align=left style="color: #777">Our motivation: (a) The distribution of singular values in spatiotemporal data is long-tailed. The existence of missing data can increase its rank (or singular values). (b) Low-rank models can filter out informative signals and generate a smooth reconstruction, resulting in truncating too much energy in the left part of its spectrum. (c) Deep models can preserve high-frequency noise and generate sharp imputations, maintaining too much energy for the right part of the singular spectrum. With the generality of low-rank models and the expressivity of deep models, ImputeFormer achieves a signal-noise balance for accurate imputation.</p>
</div>

---



## Google Scholar
**Due to the minor change of ImputeFormer's title, you can simply search for "ImputeFormer" in Google Scholar to get our latest version. Citation information is automatically updated as the proceedings become available.**

## Bibtex reference

If you find this code useful please consider to cite our paper:

```
@article{nie2023imputeformer,
  title={ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation},
  author={Nie, Tong and Qin, Guoyang and Ma, Wei and Mei, Yuewen and Sun, Jian},
  journal={arXiv preprint arXiv:2312.01728},
  year={2023}
}
```

## Acknowledgement

We acknowledge [SPIN](https://github.com/Graph-Machine-Learning-Group/spin) for providing a useful benchmark tool and [TorchSpatiotemporal](https://github.com/TorchSpatiotemporal) for model implementations.
