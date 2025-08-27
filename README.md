# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics

The dataset is available at [https://huggingface.co/datasets/weidawang/CMPhysBench](https://huggingface.co/datasets/weidawang/CMPhysBench).

## Acknowledgement
**CMPhysBench** was inspired by previous dataset works including **[PHYBench](https://www.phybench.cn/)**, **[PhysBench](https://physbench.github.io/)**, **[GPQA](https://github.com/idavidrein/gpqa)** and  **[SuperGPQA](https://supergpqa.github.io)**.

**Scalable Expression Edit Distance (SEED)** is inspired by `Expression Edit Distance (EED)` metric from **[PHYBench](https://www.phybench.cn/)**, which introduced Edit Distance to evaluating symbolic reasoning in physics. We extend and modify this idea by proposing the , supporting more diverse answer types and providing finer-grained evaluation dedicated for the fields of Condensed Matter Physics.

We sincerely thank the PHYBench team for their open-source contribution. Their code is released under the [MIT license](https://github.com/phybench-official/phybench?tab=MIT-1-ov-file#readme) and is available at [https://github.com/phybench-official/phybench](https://github.com/phybench-official/phybench).

## Citations

```bibtex
@misc{wang2025cmphysbenchbenchmarkevaluatinglarge,
      title={CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics}, 
      author={Weida Wang and Dongchen Huang and Jiatong Li and Tengchao Yang and Ziyang Zheng and Di Zhang and Dong Han and Benteng Chen and Binzhao Luo and Zhiyu Liu and Kunling Liu and Zhiyuan Gao and Shiqi Geng and Wei Ma and Jiaming Su and Xin Li and Shuchen Pu and Yuhan Shui and Qianjia Cheng and Zhihao Dou and Dongfei Cui and Changyong He and Jin Zeng and Zeke Xie and Mao Su and Dongzhan Zhou and Yuqiang Li and Wanli Ouyang and Yunqi Cai and Xi Dai and Shufei Zhang and Lei Bai and Jinguang Cheng and Zhong Fang and Hongming Weng},
      year={2025},
      eprint={2508.18124},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.18124}, 
}

@inproceedings{rein2024gpqa,
  title={Gpqa: A graduate-level google-proof q\&a benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  booktitle={First Conference on Language Modeling}
}

@article{du2025supergpqa,
  title={Supergpqa: Scaling llm evaluation across 285 graduate disciplines},
  author={Du, Xinrun and Yao, Yifan and Ma, Kaijing and Wang, Bingli and Zheng, Tianyu and Zhu, King and Liu, Minghao and Liang, Yiming and Jin, Xiaolong and Wei, Zhenlin and others},
  journal={arXiv preprint arXiv:2502.14739},
  year={2025}
}

@misc{chow2025physbenchbenchmarkingenhancingvisionlanguage,
      title={PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding}, 
      author={Wei Chow and Jiageng Mao and Boyi Li and Daniel Seita and Vitor Guizilini and Yue Wang},
      year={2025},
      eprint={2501.16411},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.16411}, 
}

@article{qiu2025phybench,
  title={Phybench: Holistic evaluation of physical perception and reasoning in large language models},
  author={Qiu, Shi and Guo, Shaoyang and Song, Zhuo-Yang and Sun, Yunbo and Cai, Zeyu and Wei, Jiashen and Luo, Tianyu and Yin, Yixuan and Zhang, Haoxu and Hu, Yi and others},
  journal={arXiv preprint arXiv:2504.16074},
  year={2025}
}

```
