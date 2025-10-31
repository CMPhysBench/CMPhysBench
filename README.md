# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics

[![Paper](https://img.shields.io/badge/Paper-B31B1B?logo=arxiv)](https://arxiv.org/abs/2508.18124)&nbsp;&nbsp;&nbsp;[![Code](https://img.shields.io/badge/Code-8A2BE2?logo=github)](https://github.com/CMPhysBench/CMPhysBench)&nbsp;&nbsp;&nbsp;[![Data](https://img.shields.io/badge/Data-00D70F?logo=huggingface)](https://huggingface.co/datasets/weidawang/CMPhysBench)&nbsp;&nbsp;&nbsp;[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/CMPhysBench/CMPhysBench/blob/main/LICENSE)


> The dataset is available at [https://huggingface.co/datasets/weidawang/CMPhysBench](https://huggingface.co/datasets/weidawang/CMPhysBench).

CMPhysBench is a benchmark for evaluating large language models in condensed matter physics, featuring 520 graduate-level calculation problems curated from standard textbooks across magnetism, superconductivity, strongly correlated systems, semiconductors, and theoretical foundations. We introduce the Scalable Expression Edit Distance (SEED) metric, which provides fine-grained partial credit for more accurate assessment of reasoning. Experiments show that even state-of-the-art models like Grok-4 achieve less than 30% accuracy, highlighting significant gaps in LLM capabilities for advanced physics reasoning.

<div align="center">
  <img src="imgs/CMPhysBench.png" width="1000"/>
</div>

## Run Evaluation

1. **Environment**

   ```bash
   pip install torch datasets tqdm vllm
   pip install sympy numpy latex2sympy2_extended timeout_decorator pint
   ```

2. **Usage**

   To run the evaluation, use the following command, specifying the path to your model:

   ```bash
   python evaluation.py --model-path /path/to/your/model --k 1
   ```

3. **Output Structure**

   The script creates a unique, timestamped directory for each run inside ``./output``. The dataset is automatically cached locally to a ``CMPhysBench`` directory to avoid re-downloading.

   ```text
   .
   ├── evaluation.py
   ├── SEED/
   ├── CMPhysBench/              # <-- Local dataset cache
   └── output/
       └── <model_name>_pass@<k>_<timestamp>/  # <-- Unique directory for each run
           ├── run.log                         # Full run log (includes final results table)
           ├── <model_name>-pass@<k>.jsonl     # Raw model inference output
           └── <model_name>-pass@<k>_final_results.json  # Final comprehensive results file with all info
   ```

## Acknowledgement

**CMPhysBench** was inspired by previous dataset works including [PHYBench](https://www.phybench.cn/), [PHYSICS](https://arxiv.org/pdf/2506.00022), [GPQA](https://github.com/idavidrein/gpqa) and  [OlympiadBench](https://github.com/OpenBMB/OlympiadBench).

**Scalable Expression Edit Distance (SEED)** is inspired by `Expression Edit Distance (EED)` metric from [PHYBench](https://www.phybench.cn/), which introduced Edit Distance to evaluating symbolic reasoning in physics. We extend and modify this idea by proposing the SEED score, supporting more diverse answer types and providing fine-grained and more robust evaluation dedicated for the fields of Condensed Matter Physics.

We sincerely thank the PHYBench team for their open-source contribution. Their code is released under the [MIT license](https://github.com/phybench-official/phybench?tab=MIT-1-ov-file#readme) and is available at [https://github.com/phybench-official/phybench](https://github.com/phybench-official/phybench).

## Citations

```bibtex
@article{wang2025cmphysbench,
  title={CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics},
  author={Wang, Weida and Huang, Dongchen and Li, Jiatong and Yang, Tengchao and Zheng, Ziyang and Zhang, Di and Han, Dong and Chen, Benteng and Luo, Binzhao and Liu, Zhiyu and others},
  journal={arXiv preprint arXiv:2508.18124},
  year={2025}
}
```
