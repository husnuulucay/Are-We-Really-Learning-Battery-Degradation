# Reassessing Battery State of Health Estimation: The Impact of True Health History on Recursive Prediction

This repository contains the official code for the paper:  
**"REASSESSING BATTERY STATE OF HEALTH ESTIMATION: THE IMPACT OF TRUE HEALTH HISTORY ON RECURSIVE PREDICTION"**

The study investigates the dependence of machine learning and deep learning models on true State of Health (SOH) lags in lithium-ion battery SOH estimation, and the performance loss under recursive deployment conditions where this access is unavailable.

## 📄 Abstract

Accurate State-of-Health (SOH) estimation is critical for lithium-ion battery safety and lifecycle management. Many machine learning studies report exceptional performance (R² > 0.95), but it remains unclear whether this comes from learning physical degradation or from temporal autocorrelation.

Using two public datasets — NASA PCoE (LCO, 4 batteries) and Severson CH17 (LFP, 1 battery) — we compare:
- **XGBoost**, **LSTM**, and **GRU** models under:
  - **Standard scenario** (true SOH lags available)
  - **Recursive scenario** (no ground truth SOH history — realistic deployment)
- **Persistence model** (predicts previous observation as current)
- **Lag-free ablation** (physical features only)

Key findings:
- Recursive prediction increases error across all learning-based models (e.g., XGBoost RMSE: 0.0189 → 0.0364; LSTM: 0.0469 → 0.0579; GRU: 0.0320 → 0.0437 on NASA PCoE).
- The Persistence model achieves the lowest error on both datasets (NASA PCoE ≈ 0.0089; CH17 = 0.0012).
- SHAP analysis shows lag features dominate model decisions (85.60% NASA PCoE; 97.51% CH17).
- The performance gap between standard and recursive evaluation reflects not the pure effect of autoregressive bias, but a combination of lag access loss, exposure bias, error feedback, and initial condition effects.

📁 Two Independent Codebases
This repository contains two independent codebases, each corresponding to a different battery chemistry:

Codebase	Battery Chemistry	Dataset
code_lco/	Li-ion (LCO)	NASA PCoE (B0005, B0006, B0007, B0018)
code_lfp/	LiFePO₄ (LFP)	Severson CH17
Both codebases can be run independently to reproduce the results for each dataset separately. They share the same model architectures (XGBoost, LSTM, GRU, Persistence) and evaluation protocols (standard, recursive, ablation), but operate on different data sources.

How to Run? 

Li-ion (LCO) — NASA PCoE:
cd code_lco
python run_experiments_lco.py

LiFePO₄ (LFP) — CH17:
cd code_lfp
python run_experiments_lfp.py


## 📦 Requirements

- Python 3.8 or higher

Required libraries (see `requirements.txt`):
- numpy
- pandas
- scikit-learn
- xgboost
- tensorflow / keras (for LSTM and GRU)
- shap
- matplotlib
- seaborn
- scipy
- statsmodels

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/husnuulucay/Are-We-Really-Learning-Battery-Degradation.git
cd battery-soh-autoregressive-bias
pip install -r requirements.txt