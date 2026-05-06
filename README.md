# Precise, Grade-Specific Text Simplification via Weighted Adapter Merger

**Master's Thesis - MSc Language Science and Technology**  
*Saarland University, March 2026*

**Author:** William LaCroix  
**Supervisors:** Prof. Dr. Alexander Koller, Dr. Ji-Ung Lee, Sarubi Thillainathan  
**Chair for Computational Linguistics**

---

## Abstract

This thesis investigates whether modular adapter composition can enable controllable readability in neural sentence simplification for second language learners. While educational reading materials must align with learner proficiency, most simplification systems treat complexity reduction as a binary task rather than targeting specific difficulty levels.

This work trains multiple low-rank adapters (LoRA) on sentence simplification data labeled by readability grade level, then combines these grade-specific adapters via weighted parameter merging to approximate intermediate difficulty levels without training a separate model for every possible target grade.

**Key Findings:**
- Grade-specific adapters capture meaningful differences in linguistic complexity but only approximate their intended readability targets
- Even the strongest models remain roughly one grade level away from the target on average
- Adapter merging allows approximate interpolation between adjacent levels but does not outperpose single-adapter fine-tuned models
- Merged configurations produce unstable behavior under many merge configurations

**Conclusion:** Adapter merging enables modular recombination of simplification models but does not provide the precise readability control required for inter-grade alignment in text simplification for language learners.

---

## Repository Structure

```
thesis-project/
├── experiments/
│   ├── configs/                   # YAML configuration files
│   ├── inspection_logs/           # Model inspection and validation logs
│   ├── logs/                      # Training logs
│   ├── preprocessing/             # Data preprocessing scripts
│   ── scripts/                   # Training and experimental scripts
├── results/
├── src/
│   ├── custom_modules/            # Custom model components
│   │   └── scripts/               # Module-specific scripts
│   └── patches/
│       └── LLaMA-Factory/         # Forked LLaMA-Factory modifications
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
note: datasets not uploaded to Github. Experiments download cleaned data subsets during training/eval from HuggingFace datasets.
---

## Research Questions

**RQ1:** Do sentence simplification adapters trained on discrete grade levels reliably generate output that corresponds to their target grade level, as measured by standard readability metrics?

**RQ2:** Can combining grade-specific adapters through weighted parameter merging improve the system's ability to generate text at intended readability levels?

**RQ3:** Can a learnable weighting scheme over grade-specific adapters enable the generation of text at arbitrary intermediate readability levels not explicitly represented in training data?

---

## Methodology

### Model Architecture
- **Base Model:** LLaMA3.2-3B-Instruct
- **Fine-Tuning:** Low-Rank Adaptation (LoRA)
- **Training Strategy:** Two-stage approach
  1. Shared warm-up model on all simplification data
  2. Grade-specific adapters (Grades 2-12) from warm-up initialization

### Dataset
- **Source:** WikiLarge corpus (preprocessed)
- **Annotation:** Flesch-Kincaid Grade Level (FKGL) labels
- **Task:** Sentence-level simplification

### Adapter Merging
- **Primary Method:** DARE-TIES (Domain-Aware Representation Enhancement with Task-Informed Embedding Similarity)
- **Strategies Evaluated:**
  - Linear weighted averaging
  - SVD-based compression
  - Magnitude-based pruning
  - Concatenation

### Evaluation Metrics
- **Readability:** Flesch-Kincaid Grade Level (FKGL)
- **Fluency:** Perplexity
- **Simplification Quality:** SARI, BERTScore F1
- **Target Alignment:** Grade-level deviation

---

## Key Results

### Baseline Performance
- Standard simplification models lack fine-grained readability control
- Binary simplification (complex → simple) insufficient for pedagogical applications

### Grade-Specific Adapters
- Individual adapters capture meaningful complexity differences
- Average deviation: ~1 grade level from target
- Best performance on mid-range grades (6-9)
- Degraded performance on extreme grades (2-3, 11-12)

### Adapter Merging
- Approximate interpolation between adjacent grades possible
- No substantial improvement over best single adapters
- High variance in merged configurations
- DARE-TIES outperforms other merge strategies but remains imprecise

### Limitations
- Readability control remains coarse rather than precise
- FKGL-only evaluation may not capture pedagogical suitability
- Sentence-level simplification ignores discourse coherence
- English-only experiments limit generalizability

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.10+
# CUDA-capable GPU (recommended: 24GB+ VRAM)
# Conda or virtualenv

conda create -n text-simplification python=3.10
conda activate text-simplification
```

### Dependencies
```bash
pip install -r requirements.txt

# Core dependencies:
# - transformers>=4.35.0
# - peft>=0.7.0
# - torch>=2.1.0
# - datasets>=2.14.0
# - evaluate>=0.4.0
# - sentencepiece
# - textstat
# - bert-score
```

### Dataset Preparation
```bash
*rewrite after full reorganization*
```

---

## Usage

### Training Grade-Specific Adapters

```bash
*rewrite after full reorganization*
```

### Merging Adapters

```bash
# Merge adapters with DARE-TIES
*rewrite after full reorganization*
```

### Inference

```bash
*rewrite after full reorganization*
```

### Evaluation

```bash
# Evaluate outputs
*rewrite after full reorganization*
```

---

## Computational Requirements

### Training
- **Hardware:** Single NVIDIA A100 (40GB) or equivalent
- **Time per adapter:** ~6-8 hours
- **Storage:** ~5GB per grade-specific adapter
- **Total training time:** ~80-100 hours for all adapters (Grades 2-12)

### Inference
- **Batch size:** 8-16 (depending on GPU memory)
- **Throughput:** ~50-100 sentences/minute

### HTCondor Configuration
For cluster-based training, see `experiments/condor_jobs/` for submit files and resource specifications.

---

## Reproducing Results

### Full Experimental Pipeline

```bash
*rewrite after full reorganization*
```

### Random Seeds
All experiments use fixed random seeds for reproducibility:
- NumPy: 42
- PyTorch: 42
- Transformers: 42

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{lacroix2026GradeSpecificTextSimplification,
  title={Precise, Grade-Specific Text Simplification via Weighted Adapter Merger for Second Language Acquisition},
  author={LaCroix, William},
  year={2026},
  school={Saarland University},
  type={Master's Thesis},
  address={Saarbr{\"u}cken, Germany}
}
```

---

## Related Work

This thesis builds on and contributes to the following research areas:

### Controllable Text Simplification
- Martin et al. (2020) - ACCESS: Controllable Sentence Simplification
- Agrawal & Carpuat (2023) - Grade-level conditioning with T5
- Thillainathan & Koller (2024) - Fine-grained control through in-context learning

### Model Merging
- Wortsman et al. (2022) - Model Soups
- Li et al. (2022) - Branch-Train-Merge
- Sukhbaatar et al. (2024) - Branch-Train-Mix
- Yadav et al. (2023) - TIES merging
- Yu et al. (2024) - DARE

### Text Simplification for L2 Learning
- Crossley et al. (2008) - Readability in SLA
- Nation (2001) - Vocabulary learning
- Rodrigo (2016) - Graded readers validation

---

## License

This research code is released under the MIT License. See `LICENSE` for details.

The WikiLarge dataset is used under its original license.

---

## Acknowledgments

Special thanks to:
- Dr. Ji-Ung Lee and Sarubi Thillainathan for thoughtful guidance throughout this project
- Prof. Dr. Alexander Koller for years of mentorship

---

## Contact

**William LaCroix**  
Master's Student, Language Science and Technology  
Saarland University  

For questions about this research, please open an issue in this repository.

---

## Future Directions

Potential extensions of this work:

1. **Multi-metric optimization:** Incorporate linguistic features beyond FKGL
2. **Human evaluation:** Conduct comprehension studies with L2 learners
3. **Cross-lingual extension:** Test approach on non-English simplification
4. **Document-level coherence:** Extend beyond sentence-level simplification
5. **Curriculum alignment:** Map to pedagogical frameworks (CEFR, etc.)
6. **Dynamic adapter selection:** Runtime selection based on input complexity

---

**Last Updated:** March 2026
