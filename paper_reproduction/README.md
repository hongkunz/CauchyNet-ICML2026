# Paper-ordered reproduction map

The scripts follow the numbering in `paper/CauchyNet_ICML2026.pdf`.

| Paper item | Script | Evidence |
|---|---|---|
| Figure 1 | `01_figure1_rational_spike.py` | seeded protocol reconstruction |
| Figure 2 | `02_figure2_cauchy_activation.py` | exact formula and parameters |
| Figure 3 | `03_figure3_theory_roadmap.py` | deterministic diagram |
| Figure 4 | `04_figure4_elliptical_poles.py` | exact initialization geometry |
| Figure 5 | `05_figure5_gap_filling.py` | exact saved arrays and plotter |
| Figure 6 | `06_figure6_architecture_summary.py` | deterministic diagram |
| Figure 7 | `07_figure7_gap_filling_curves.py` | exact 10-seed arrays |
| Figure 8 | `08_figure8_missing_disk.py` | final released reconstruction |
| Figure 9 | `09_figure9_2d_surface.py` | verified 20,000-epoch protocol |
| Figure 10 | `10_figure10_m4_decomposition.py` | exact embedded M4 data |
| Figure 11 | `11_figure11_targets.py` | exact formulas and embedded data |
| Figure 12 | `12_figure12_m4_benchmark.py` | seeded M4 reconstruction |
| Figure 13 | `13_figure13_predictions_residuals.py` | seeded M4 reconstruction |
| Figure 14 | `14_figure14_imaginary_penalty.py` | seeded trainable-pole ablation |
| Table 1 | `16_table1_gap_filling.py` | exact per-seed summary |
| Table 2 | `17_table2_notation.py` | deterministic notation table |
| Table 3 | `18_table3_default_setup.py` | deterministic setup table |
| Table 4 | `19_table4_parameter_counts.py` | instantiated parameter counts |
| Table 5 | `20_table5_fixed_pole_ridge.py` | exact closed-form coefficient fitting |
| Table 6 | `21_table6_uci_configurations.py` | exact saved configurations |
| Table 7 | `22_table7_uci_results.py` | exact saved five-fold summaries |
| UCI producer | `23_experiment12_uci_tuned_sweep.py` | bounded replacement sweep |

## Important protocol notes

- Figure 9 uses seed 10, width 128, 300 points split 150/75/75,
  training-only target scaling, and 20,000 epochs. Its shipped result reports
  test MSE `1.3228622037786408e-06` and dense-grid error
  `[-0.0055606365, 0.0056686401]`.
- Figure 14 reports absolute normalized test error on a linear axis.
- Table 4's Informer count of 199,553 uses two encoder layers.
- Table 5 fixes the pole dictionary and fits only the output coefficients by
  closed-form ridge least squares.
- Figures 10--13 use the compact M4 H1 payload embedded in `_m4_data.py`.
- The removed sensitivity Figure 15 and all of its tuning outputs are absent
  from this public package.

All generated files are written under `outputs/figures` and
`outputs/results`. The shipped figure directory contains only the 14 final
vector PDFs. Scripts may generate PNG previews locally, but those previews are
ignored by Git.
