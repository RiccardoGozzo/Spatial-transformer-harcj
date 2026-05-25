# Spatial Transformer HAR-CJ
 
> Realized volatility forecasting with endogenous sparse cross-sectional spillovers.
 
This repository implements a rolling realized-volatility forecasting pipeline that combines a pooled **HAR-CJ econometric anchor** with a **Spatial Transformer** correction learned via single-head sparse attention. The project extends and re-analyses the work of Zhang, Pu, Cucuringu and Dong (2025).
 
---
 
## Based on
 
> **Zhang, C., Pu, X., Cucuringu, M., & Dong, X. (2025).**  
> *Forecasting realized volatility with spillover effects: Perspectives from graph neural networks.*  
> Journal of Financial Econometrics, nbae026.  
> [https://doi.org/10.1093/jjfinec/nbae026](https://doi.org/10.1093/jjfinec/nbae026)
 
The paper introduces the **GHAR** and **GNNHAR** models, which augment the standard HAR structure with cross-sectional volatility spillovers modelled via a financial network. This repository replicates the core results and proposes a further extension replacing the GLASSO adjacency matrix with an endogenous sparse attention graph.
 
---
 
## Architecture
 
The forecast is a multiplicative correction of the econometric anchor:
 
```
RV_hat[i,t] = HAR_CJ[i,t] * M[i,t],    M[i,t] ∈ [0.80, 1.50]
```
 
The correction `M[i,t]` is learned by a **Spatial Transformer** that:
 
- Builds a directed latent graph over N assets via **single-head Sparsemax attention** (exact zeros for weak connections)
- Uses a **learnable temperature** τ to control graph sparsity end-to-end
- Aggregates cross-sectional spillover information through a residual attention block + FFN
- Applies a **quantile OOD gate** that shrinks the correction when lagged volatility enters empirical tail states
### Key design choices
 
| Choice | Motivation |
|---|---|
| Single attention head | Multi-head averaging inflates graph degree; single head preserves sparsity |
| Sparsemax | Exact zeros for weak edges, no post-hoc thresholding |
| Learnable temperature τ = exp(log τ) | Allows both sparser (τ > 1) and denser (τ < 1) graphs |
| IPR penalty − λ Σ A²ᵢⱼ | Scale-invariant sparsity regulariser in [1/N, 1]; more principled than entropy |
| Pooled HAR-CJ anchor | Estimated separately via QLIKE / L-BFGS-B; network only learns the correction |
 
---
 
## Model hierarchy
 
```
HAR  →  GHAR  →  GNNHAR  →  GHAR-CJ  →  Spatial Transformer HAR-CJ
```
 
Each step adds one contribution: cross-sectional spillovers, nonlinear GNN correction, richer HAR baseline, endogenous sparse graph.
 
---
 
## Results
 
| Model | MSE | MAE | QLIKE | DA | MCS |
|---|---|---|---|---|---|
| HAR | 1.000 | 1.000 | 1.000 | 1.000 | — |
| GHAR | 0.988 | 0.993 | 0.994 | 1.012 | — |
| GNNHAR | 0.983 | 0.989 | 0.993 | 1.030 | — |
| HAR-CJ | 0.980 | 0.984 | 0.994 | 1.025 | — |
| GHAR-CJ | 0.972 | 0.980 | 0.989 | 1.031 | - |
| **Spatial Transformer HAR-CJ** | **0.943** | **0.961** | **0.981** | **1.054** | **✓** |
 
Performance ratios relative to HAR. Values below 1 indicate improvement for MSE/MAE/QLIKE; above 1 for DA.
 
---
 
## Graph properties (Spatial Transformer)
 
| Diagnostic | Value |
|---|---|
| Hard degree (non-zero edges) | 2.1 |
| Effective degree (IPR-based) | 10.2 |
| Sparsity (exact zeros) | 94.4% |
| IPR | 0.740 |
| Learned temperature τ | 2.20 |
| Attention entropy (normalised) | 0.123 |
 
---
 
## Repository structure
 
```
├── spatial_transformer_final.py   # Main pipeline (data prep, model, rolling loop, evaluation)
├── asset_heterogeneity_cell.py    # Per-asset performance analysis
└── README.md
```
 
---
 
## Data
 
Realized volatility is sourced from the **VOLARE** database:
 
> Cipollini, F., Cruciani, G., Gallo, G.M., Insana, A., Otranto, E., & Spagnolo, F. (2026).  
> *Volatility Archive for Realized Estimates (VOLARE).*  
> arXiv:2602.19732.
 
The database is **not included** in this repository. Data covers January 2015 – March 2026 using a subsampled 5-minute RV estimator.
 
---
 
## Usage
 
```python
from spatial_transformer_final import run_example
 
# df_filtered : long-format DataFrame with columns date, symbol, rv5_ss, bv5_ss
# log_returns : wide-format DataFrame of daily log-returns
 
risultati = run_example(df_filtered, log_returns)
```
 
Key parameters in `run()`:
 
```python
run(
    df_filtered, log_returns,
    init_temperature = 2,   # < 1 → denser graph; > 1 → sparser
    lambda_edges     = 0.05,  # IPR sparsity weight
    test_threshold   = 0.02,  # hard pruning threshold for graph analysis
)
```
 
---
 
## Requirements
 
```
torch
numpy
pandas
scipy
scikit-learn
networkx
matplotlib
tqdm
```
 
---
## Author
 
**Riccardo Gozzo** — June 2026
 
