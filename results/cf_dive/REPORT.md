# CF-DIVE experiment results

Each dataset uses one fixed transductive split. The paired intervals describe acquisition and model seed variability on that split. Historical pretraining, cache creation, process startup and result serialization are excluded from the timing.

## citeseer

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.0, 0.15]. Best zero-CF bonus 0.15.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 64.42 ± 1.36 | 64.82 ± 1.07 | 64.82 ± 1.07 |
| 0.1 | 61.32 ± 4.47 | 62.28 ± 5.18 | 62.28 ± 5.18 |
| 0.3 | 64.33 ± 3.82 | 63.54 ± 4.30 | 63.54 ± 4.30 |
| 0.6 | 63.88 ± 4.87 | 62.86 ± 4.82 | 63.28 ± 4.39 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 53.45 | 60.82 | 60.41 | 64.67 | 60.09 ± 4.75 | 58.29 |
| entropy | 53.45 | 59.34 | 61.76 | 66.14 | 60.30 ± 5.57 | 60.49 |
| disagreement | 53.45 | 60.66 | 61.72 | 65.64 | 60.64 ± 4.71 | 60.11 |
| no_cf | 53.45 | 60.69 | 64.39 | 67.30 | 61.82 ± 4.80 | 61.05 |
| tuned_no_cf | 53.45 | 60.69 | 64.36 | 67.34 | 61.81 ± 4.79 | 61.09 |
| cf | 53.45 | 63.57 | 66.55 | 67.96 | 63.61 ± 2.73 | 62.29 |
| cf_top | 53.45 | 62.70 | 65.05 | 65.77 | 62.45 ± 3.75 | 59.24 |
| cf_milp | 53.45 | 63.57 | 66.55 | 67.96 | 63.61 ± 2.73 | 62.29 |
| raw_ensemble_cf | 56.90 | 62.88 | 67.15 | 67.18 | 64.02 ± 2.62 | 61.45 |
| single_cf | 55.61 | 62.10 | 65.77 | 67.15 | 63.08 ± 2.32 | 61.70 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +4.00, +4.39, -0.63, -1.54, +2.74. Mean +1.79, descriptive paired t interval [-1.58, +5.16].

Across 2,055 independent edge probes in the CF arm, the average member flipped at the target in 16.84% and at the other endpoint in 9.25%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.029 | 0.059 | 0.090 | 0.120 |
| entropy | 0.029 | 0.059 | 0.090 | 0.120 |
| disagreement | 0.028 | 0.059 | 0.089 | 0.120 |
| no_cf | 0.028 | 0.062 | 0.093 | 0.124 |
| tuned_no_cf | 0.029 | 0.060 | 0.091 | 0.122 |
| cf | 0.029 | 0.063 | 0.096 | 0.129 |
| cf_top | 0.029 | 0.064 | 0.097 | 0.129 |
| cf_milp | 0.028 | 0.064 | 0.100 | 0.136 |
| raw_ensemble_cf | 0.028 | 0.061 | 0.094 | 0.126 |
| single_cf | 0.015 | 0.032 | 0.049 | 0.066 |

Summed validation and test trial time 0.0029 device-hours. One-time preparation 0.19 seconds. Device: NVIDIA GeForce RTX 5070.

## cora

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.3, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 65.74 ± 3.84 | 65.74 ± 3.84 | 65.74 ± 3.84 |
| 0.1 | 67.58 ± 4.96 | 67.63 ± 4.89 | 67.63 ± 4.89 |
| 0.3 | 68.18 ± 3.88 | 67.79 ± 3.65 | 67.84 ± 3.54 |
| 0.6 | 66.20 ± 2.30 | 65.96 ± 1.79 | 67.79 ± 3.65 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 63.87 | 67.64 | 70.96 | 74.24 | 69.22 ± 3.42 | 67.97 |
| entropy | 63.87 | 68.45 | 71.33 | 73.36 | 69.46 ± 1.43 | 69.67 |
| disagreement | 63.87 | 65.57 | 69.23 | 70.11 | 67.26 ± 3.63 | 63.06 |
| no_cf | 63.87 | 65.42 | 69.89 | 73.43 | 67.99 ± 1.74 | 69.13 |
| tuned_no_cf | 63.87 | 65.42 | 69.89 | 73.43 | 67.99 ± 1.74 | 69.13 |
| cf | 63.87 | 67.82 | 73.47 | 74.54 | 70.17 ± 1.81 | 70.89 |
| cf_top | 63.87 | 68.12 | 72.69 | 75.02 | 70.09 ± 1.66 | 71.11 |
| cf_milp | 63.87 | 67.82 | 73.47 | 74.54 | 70.17 ± 1.81 | 70.89 |
| raw_ensemble_cf | 68.52 | 73.06 | 75.57 | 76.35 | 73.69 ± 2.71 | 71.42 |
| single_cf | 68.19 | 71.55 | 73.36 | 77.23 | 72.54 ± 2.20 | 72.46 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +0.43, +2.12, +6.40, +1.72, +0.22. Mean +2.18, descriptive paired t interval [-0.92, +5.28].

Across 3,423 independent edge probes in the CF arm, the average member flipped at the target in 16.90% and at the other endpoint in 7.58%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.028 | 0.058 | 0.088 | 0.117 |
| entropy | 0.029 | 0.059 | 0.090 | 0.121 |
| disagreement | 0.029 | 0.059 | 0.089 | 0.119 |
| no_cf | 0.029 | 0.060 | 0.090 | 0.121 |
| tuned_no_cf | 0.029 | 0.060 | 0.090 | 0.121 |
| cf | 0.028 | 0.061 | 0.093 | 0.126 |
| cf_top | 0.029 | 0.063 | 0.096 | 0.128 |
| cf_milp | 0.028 | 0.063 | 0.098 | 0.134 |
| raw_ensemble_cf | 0.028 | 0.060 | 0.092 | 0.124 |
| single_cf | 0.015 | 0.032 | 0.050 | 0.067 |

Summed validation and test trial time 0.0035 device-hours. One-time preparation 0.38 seconds. Device: NVIDIA GeForce RTX 5070.

## pubmed

Positive-CF validation choice γ=0.1, coverage bonus=0.05. Best unrestricted validation pair [0.1, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 72.94 ± 4.35 | 72.82 ± 4.56 | 72.82 ± 4.56 |
| 0.1 | 74.62 ± 2.57 | 74.27 ± 2.98 | 74.27 ± 2.98 |
| 0.3 | 72.72 ± 4.27 | 72.39 ± 4.31 | 72.39 ± 4.31 |
| 0.6 | 72.69 ± 4.24 | 72.81 ± 4.24 | 72.55 ± 4.23 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 64.41 | 73.65 | 75.90 | 76.85 | 73.39 ± 2.18 | 76.62 |
| entropy | 64.41 | 73.92 | 77.17 | 78.27 | 74.14 ± 2.33 | 78.10 |
| disagreement | 64.41 | 68.74 | 71.64 | 72.20 | 69.56 ± 5.40 | 71.57 |
| no_cf | 64.41 | 72.77 | 72.61 | 74.86 | 71.67 ± 2.78 | 74.61 |
| tuned_no_cf | 64.41 | 72.77 | 72.61 | 74.86 | 71.67 ± 2.78 | 74.61 |
| cf | 64.41 | 72.59 | 73.66 | 75.44 | 72.05 ± 3.67 | 75.13 |
| cf_top | 64.41 | 71.28 | 71.40 | 73.76 | 70.59 ± 3.87 | 73.75 |
| cf_milp | 64.41 | 72.59 | 73.66 | 75.44 | 72.05 ± 3.67 | 75.13 |
| raw_ensemble_cf | 65.99 | 70.47 | 75.72 | 77.09 | 72.58 ± 1.17 | 76.84 |
| single_cf | 65.27 | 71.64 | 75.02 | 76.63 | 72.54 ± 1.19 | 76.43 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): +1.27, -1.21, -2.18, +3.19, +0.85. Mean +0.38, descriptive paired t interval [-2.25, +3.02].

Across 3,811 independent edge probes in the CF arm, the average member flipped at the target in 15.36% and at the other endpoint in 9.60%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.033 | 0.068 | 0.103 | 0.138 |
| entropy | 0.032 | 0.067 | 0.104 | 0.140 |
| disagreement | 0.035 | 0.072 | 0.108 | 0.145 |
| no_cf | 0.034 | 0.074 | 0.110 | 0.148 |
| tuned_no_cf | 0.034 | 0.071 | 0.108 | 0.148 |
| cf | 0.037 | 0.077 | 0.117 | 0.161 |
| cf_top | 0.035 | 0.075 | 0.117 | 0.159 |
| cf_milp | 0.036 | 0.079 | 0.121 | 0.166 |
| raw_ensemble_cf | 0.033 | 0.075 | 0.117 | 0.160 |
| single_cf | 0.017 | 0.040 | 0.062 | 0.084 |

Summed validation and test trial time 0.0037 device-hours. One-time preparation 1.43 seconds. Device: NVIDIA GeForce RTX 5070.

## wikics

Positive-CF validation choice γ=0.3, coverage bonus=0.05. Best unrestricted validation pair [0.0, 0.05]. Best zero-CF bonus 0.05.

Validation AULC in percent, mean ± sample SD over three seeds.

| γ / bonus | .05 | .15 | .30 |
| --- | --- | --- | --- |
| 0.0 | 58.39 ± 2.49 | 58.39 ± 2.49 | 58.39 ± 2.49 |
| 0.1 | 56.98 ± 7.82 | 56.97 ± 7.82 | 56.97 ± 7.82 |
| 0.3 | 57.84 ± 5.51 | 57.34 ± 7.40 | 57.38 ± 7.47 |
| 0.6 | 57.84 ± 5.51 | 57.84 ± 5.51 | 57.34 ± 7.40 |

Test accuracy by acquired labels, normalized accuracy AULC, and final macro-F1, all in percent.

| Method | 16 | 24 | 32 | 40 | AULC ± SD | F1 @ 40 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 57.04 | 61.77 | 64.79 | 66.89 | 62.84 ± 4.26 | 54.87 |
| entropy | 57.04 | 60.19 | 61.32 | 65.81 | 60.98 ± 2.87 | 53.34 |
| disagreement | 57.04 | 60.49 | 62.85 | 65.36 | 61.51 ± 4.82 | 52.18 |
| no_cf | 57.04 | 60.55 | 64.80 | 66.81 | 62.42 ± 3.63 | 57.67 |
| tuned_no_cf | 57.04 | 60.55 | 64.80 | 66.81 | 62.42 ± 3.63 | 57.67 |
| cf | 57.04 | 62.01 | 64.74 | 66.38 | 62.82 ± 2.57 | 57.45 |
| cf_top | 57.04 | 59.05 | 63.40 | 65.47 | 61.24 ± 3.30 | 55.55 |
| cf_milp | 57.04 | 62.01 | 64.74 | 66.38 | 62.82 ± 2.57 | 57.45 |
| raw_ensemble_cf | 53.41 | 55.89 | 59.62 | 61.94 | 57.73 ± 3.88 | 52.52 |
| single_cf | 52.74 | 56.02 | 59.72 | 61.74 | 57.66 ± 2.77 | 54.31 |

Paired CF minus no_cf AULC in seed order 41–45 (percentage points): -0.67, -0.19, -0.90, -1.94, +5.67. Mean +0.40, descriptive paired t interval [-3.35, +4.15].

Across 3,126 independent edge probes in the CF arm, the average member flipped at the target in 7.98% and at the other endpoint in 1.97%.

Mean cumulative wall seconds by label budget on one device. Divide by 3,600 for allocated device-hours.

| Method | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- |
| random | 0.031 | 0.065 | 0.098 | 0.130 |
| entropy | 0.030 | 0.064 | 0.097 | 0.130 |
| disagreement | 0.030 | 0.064 | 0.097 | 0.129 |
| no_cf | 0.029 | 0.062 | 0.094 | 0.126 |
| tuned_no_cf | 0.033 | 0.071 | 0.111 | 0.152 |
| cf | 0.037 | 0.083 | 0.127 | 0.170 |
| cf_top | 0.036 | 0.082 | 0.125 | 0.169 |
| cf_milp | 0.033 | 0.073 | 0.115 | 0.156 |
| raw_ensemble_cf | 0.029 | 0.064 | 0.097 | 0.131 |
| single_cf | 0.015 | 0.033 | 0.052 | 0.070 |

Summed validation and test trial time 0.0032 device-hours. One-time preparation 1.26 seconds. Device: NVIDIA GeForce RTX 5070.
