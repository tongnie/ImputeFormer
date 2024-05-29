# <div align="center">[KDD-24] ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation </div>

[![KDD](https://img.shields.io/badge/KDD-2024-blue.svg?style=flat-square)](#)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=ImputeFormer&color=red&logo=arxiv)](https://arxiv.org/abs/2312.01728)

**[Updata 20240526]: We will release the full model training and evaluation codes soon after carefully preparing the final version.**

<div align=center>
	<img src="./sparse_att.png" alt="Example of the sparse spatiotemporal attention layer."/>
	<p align=left style="color: #777">Example of the sparse spatiotemporal attention layer. On the left, the input spatiotemporal graph, with time series associated with every node. On the right, how the layer acts to update target representation (highlighted by the green box), by simultaneously performing inter-node spatiotemporal cross-attention (red block) and intra-node temporal self-attention (violet block).</p>
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
