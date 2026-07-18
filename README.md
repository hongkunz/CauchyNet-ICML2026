# CauchyNet ICML 2026: final paper reproduction

This is the minimal public reproducibility package for **CauchyNet: Compact
and Data-Efficient Learning Using Holomorphic Activation Functions**. It
contains only the camera-ready paper, the code and saved data needed for its
numbered Figures 1--14 and Tables 1--7, and the final generated outputs.

Smoke runs, tuning trials, legacy notebooks, intermediate figures, and the
removed sensitivity Figure 15 are intentionally excluded.

## Contents

```text
CauchyNet_GitHub_release/
├── paper/CauchyNet_ICML2026.pdf
├── paper_reproduction/
│   ├── 01_...py through 14_...py       # Figures 1--14
│   ├── 16_...py through 22_...py       # Tables 1--7
│   ├── 23_experiment12_...py           # UCI producer for Tables 6--7
│   ├── outputs/figures/                 # 14 final vector-PDF figures
│   └── outputs/results/                 # final JSON, NPZ, and TeX evidence
├── experiments/                         # only data/producers required by paper scripts
├── reproduce.py
├── requirements.txt
├── requirements-lock.txt
└── SOURCE_MANIFEST_SHA256.txt
```

## Installation

Python 3.10 was used for the final audit.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

## Fast verification

The following command regenerates the deterministic figures and tables from
the shipped arrays and verifies the saved numerical records:

```bash
python reproduce.py --mode verify
```

It includes the exact Table 1 gap-filling values:

| Model | Test MAE |
|---|---:|
| CauchyNet | 0.0202 |
| SIREN | 0.0872 |
| FNN | 0.2032 |
| N-BEATS | 0.1940 |

## Reproduction modes

```bash
python reproduce.py --mode figures    # regenerate Figures 1--14
python reproduce.py --mode tables     # regenerate Tables 1--7
python reproduce.py --mode all        # regenerate all figures and tables
python reproduce.py --mode uci-sweep  # rerun the Experiment-12 UCI sweep
```

`figures` and `all` are expensive because Figure 9 uses the verified 20,000-
epoch protocol. `uci-sweep` may download Diabetes, California Housing, and
Wine Quality Red if they are not already cached.

The paper-to-code map and protocol notes are in
[`paper_reproduction/README.md`](paper_reproduction/README.md).

The scripts also generate PNG previews when executed, but PNG files are
ignored by Git and are not included in this final release.

## Integrity

Every shipped file, including the paper and final outputs, is covered by the
SHA-256 manifest:

```bash
shasum -a 256 -c SOURCE_MANIFEST_SHA256.txt
```

On Linux, use `sha256sum -c SOURCE_MANIFEST_SHA256.txt`.

## Citation

```bibtex
@inproceedings{zhang2026cauchynet,
  title     = {{CauchyNet}: Compact and Data-Efficient Learning Using Holomorphic Activation Functions},
  author    = {Zhang, Hong-Kun and Li, Xin and Yang, Sikun and Xia, Zhihong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

Code is released under the MIT License.
