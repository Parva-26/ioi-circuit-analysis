# IOI Circuit Analysis: Replication + Template Stability Extension

Mechanistic interpretability study on GPT-2 small, replicating the **Indirect Object Identification (IOI)** circuit from [Wang et al. 2022](https://arxiv.org/abs/2211.00593) using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), extended with a circuit-stability probe under distractor-token prompt variations.

## What this is

GPT-2 small can reliably complete prompts like:

> *"When **John** and **Mary** went to the store, **Mary** gave a drink to ___"* → **John**

This requires the model to track which name is the Subject (repeated) and which is the Indirect Object (appears once), and route the correct one to the output. Wang et al. found this is implemented by an interpretable circuit of ~26 attention heads across three functional groups (Duplicate Token Heads, S-Inhibition Heads, Name Mover Heads).

This repo:
1. **Replicates** the core finding — Direct Logit Attribution (DLA) + path patching to identify and causally verify Name Mover Heads.
2. **Extends** it — tests whether the *same* heads remain causally necessary when an irrelevant distractor name is inserted into the sentence, as a lightweight probe into circuit robustness under distribution shift (motivated by open problems in circuit stability across contexts, adjacent to Neel Nanda's 200 Concrete Open Problems in Mechanistic Interpretability).

## Method

| Step | Technique |
|---|---|
| Dataset | ABBA-form IOI prompts (base) + distractor variant with a third named entity inserted mid-sentence |
| Metric | Logit difference: `logit(IO) - logit(S)` at the final token position |
| Attribution | Direct Logit Attribution (DLA) per attention head, projected onto the logit-diff direction via the unembedding |
| Causal verification | Path patching — patch each candidate head's output with its value from an IO/S-flipped run, measure the resulting drop in logit diff |
| Stability probe | Repeat DLA + path patching on the distractor dataset; compare head-importance rank correlation (Spearman) and causal-drop retention against the base dataset |

## Results

- `dla_heatmap.png` : Per-head DLA on the base template
  <p>
  <img src="results/dla_heatmap.png" width="60%" />
  </p>
  
- `dla_base_vs_distractor.png` : Side-by-side DLA comparison
  <p>
  <img src="results/dla_base_vs_distractor.png" width="60%" />
  </p>
  
- `attn_L*H*.png` : Attention pattern visualizations for top candidate heads
  <p>
  <img src="results/attn_L9H9.png" width="60%" />
  </p>
  <p>
  <img src="results/attn_L9H6.png" width="60%" />
  </p>
  <p>
  <img src="results/attn_L10H6.png" width="60%" />
  </p>
  
- `path_patching_base.csv` : Causal drop per head, base template
- `circuit_stability_results.csv` : Causal drop per head, base vs. distractor
- `circuit_stability_barplot.png` : Visual comparison of circuit stability
  <p>
  <img src="results/circuit_stability_barplot.png" width="60%" />
  </p>

## Summary

-Replicated the core finding that a small set of late-layer attention heads (Name Mover Heads) causally drive the IOI task's logit difference in GPT-2 small, consistent with Wang et al. (2022).

-Confirmed via Direct Logit Attribution + path patching, not just attention pattern inspection.

-Extension finding: We tested whether the same heads that drive the IOI task on the base template still matter when we add a distractor name to the prompt. The results were pretty clean, the head rankings between base and distractor prompts correlate at 0.781 (p=7.03e-31), which is really high. The model also keeps 98.2% of its performance even with the distractor added (logit diff goes from 3.161 to 3.104). This means the same Name Mover Heads stay causally important regardless of whether there's extra noise in the prompt. Basically, the circuit isn't just memorizing surface patterns, it's actually learning to distinguish S from IO structurally, which is why it generalizes. This extends Wang et al.'s work by showing the circuit is stable across different prompt structures, which is something the community cares about but hadn't been tested directly on the IOI task before.

## Setup

```bash
pip install -r requirements.txt
```

Or in Google Colab, uncomment the first `!pip install` line in the notebook.

Runs on CPU (slow) or GPU (recommended — a free Colab T4 is enough).

## Usage

Open `notebooks/ioi_circuit_analysis.ipynb` and run all cells top to bottom. Each section is self-contained and documented. Outputs are saved automatically to `results/`.

## Repo structure

```
ioi-circuit-analysis/
├── notebooks/
│   └── ioi_circuit_analysis.ipynb   # full analysis pipeline
├── results/                         # generated plots + CSVs
├── requirements.txt
└── README.md
```

## References

- Wang, K. et al. (2022). *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Nanda, N. *TransformerLens.* [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Conmy, A. et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)

## Author

Parva Mehta — [github.com/Parva-26](https://github.com/Parva-26)
