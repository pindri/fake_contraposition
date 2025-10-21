# PAG Robustness — Code for “Probably Approximately Global Robustness Certification”

This repository contains code to replicate results for the paper:

> **Probably Approximately Global Robustness Certification**  
> Peter Blohm, Patrick Indri, Thomas Gärtner, Sagar Malhotra  
> ICML 2025 (PMLR 267)  
> PMLR page: https://proceedings.mlr.press/v267/blohm25a.html

---

## Repository structure

- `experiments/` — Python experiment scripts used to produce results.
- `pag_robustness/` — Python library code for the certification logic.
- `rob/onnx_models/` — ONNX models referenced by some experiments.
- `summary/` — Supplemental materials.
- `plot_results.R` — R script for generating plots from CSV results.
- `simulate_bounds.R` — R script for simulating/visualising bounds.
- `env.yml` — Conda environment specification.

The exact experiment entry points and their arguments are defined in the scripts themselves. Use each script’s `--help` to see supported options.

---

## Installation

```bash
git clone https://github.com/pindri/fake_contraposition.git
cd fake_contraposition
conda env create -f env.yml
conda activate pag-robustness
```

---

## Running experiments

Run the appropriate Python script from `experiments/`.
This repository does not prescribe a single unified CLI; parameters (e.g., model paths, datasets, output locations) are specific to each script.

---

## Plotting results

The script `plot_results.R` reads CSV files from the results folders used by your runs (e.g., files like `results/AT_*.csv`, `results/Standard_*.csv`) and writes figures. Run it with R:

```bash
Rscript plot_results.R
```

If you need to change input/output paths, edit the variables in `plot_results.R` accordingly.

---

Adjust any parameters directly in the script as needed.

---

## Paper

- PMLR page: https://proceedings.mlr.press/v267/blohm25a.html

If you use this code, please cite:

```bibtex
@InProceedings{pmlr-v267-blohm25a,
  title     = {Probably Approximately Global Robustness Certification},
  author    = {Blohm, Peter and Indri, Patrick and G{"a}rtner, Thomas and Malhotra, Sagar},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {4570--4587},
  year      = {2025},
  editor    = {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--19 Jul},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v267/blohm25a.html}
}
```

---

## License

Released under the MIT License. See [LICENSE](./LICENSE).
