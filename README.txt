# Reassessing Battery SOH Prediction: Autoregressive Bias in Reported Performance

This repository contains the official code for the paper:  
**"REASSESSING BATTERY SOH PREDICTION: AUTOREGRESSIVE BIAS IN REPORTED PERFORMANCE"**

The study systematically investigates whether machine learning models truly learn battery degradation dynamics or merely exploit temporal autocorrelation (autoregressive bias) to achieve high reported performance.

## 📄 Abstract

Accurate State-of-Health (SOH) prediction is critical for lithium-ion battery safety and lifecycle management. Many machine learning studies report exceptional performance (R² > 0.95), but it remains unclear whether this comes from learning physical degradation or from temporal autocorrelation.

Using the NASA PCoE battery dataset, we compare:
- **Standard model** (with access to lagged SOH features)
- **Recursive model** (no ground truth SOH history — realistic deployment)
- **Persistence model** (predicts previous observation as current)

Key findings:
- Standard model: R² = 0.9536
- Recursive model: R² = 0.8092
- Persistence model: R² = 0.9895 (outperforms all ML models)

SHAP analysis confirms that lagged features dominate predictions over physical variables (voltage, temperature). This study provides methodological recommendations to avoid autoregressive bias in SOH prediction.

## 📦 Requirements

- Python 3.8 or higher

Required libraries (see `requirements.txt`):
- numpy
- pandas
- scikit-learn
- xgboost
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


Data Preparation
Download the NASA PCoE battery dataset from:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Place the following files into data/raw/:

B0005.mat

B0006.mat

B0007.mat

B0018.mat


🚀 Usage / Reproduction
To reproduce all results from the paper, run the scripts in the following order:

Step 1: Feature Extraction
python src/features/build_features.py


Extracts cycle-wise statistics (mean, std, min, max) from voltage, current, temperature

Creates rolling window features (window size = 3)

Generates adaptive degradation features

Output: data/processed/features.csv


Step 2: Train Models and Evaluate

python run_experiments.py


This script performs:

Leave-One-Battery-Out (LOBO) cross-validation

Standard model training (with lag features)

Recursive model forecasting (without ground truth)

Persistence model baseline calculation

Metric calculation (R², RMSE, MAE, MAPE)


Step 3: Horizon Analysis

python src/evaluation/horizon_analysis.py

Evaluates recursive model performance at horizons: 1, 5, 10, 20 cycles

Output: results/tables/horizon_results.csv

Step 4: SHAP Interpretability Analysis

python src/visualization/shap_analysis.py

Generates SHAP summary plots

Computes feature importance for lag vs. physical features

Output: results/figures/shap_summary.png

Step 5: Statistical Validation

python src/evaluation/statistics.py

Runs Linear Mixed Model (LMM) analysis

Calculates Cohen's d effect sizes

Performs Wilcoxon signed-rank tests

Output: results/tables/statistical_results.csv

Step 6: Generate All Figures

python src/visualization/plot_results.py

Produces all figures from the paper:

1_r2_comparison.png — R² comparison

2_rmse_comparison.png — RMSE comparison

3_horizon_performance.png — Horizon-based performance

4_error_propagation.png — Error propagation rates

5_actual_vs_predicted_B0005.png to 8_actual_vs_predicted_B0018.png — Time series plots

10_heatmap_rmse.png — RMSE heatmap

📊 Expected Outputs
After running all scripts, the results/ directory will contain:

results/
├── figures/          # All paper figures (PNG format)
├── tables/           # Performance metrics (CSV format)
└── models/           # Trained model checkpoints (optional)

Key results to expect:

Persistence model outperforms all ML approaches

Recursive model shows significant performance drop (especially for B0007)

Lag features dominate SHAP importance

Horizon analysis shows R² drops from 0.80 to 0.48 at 20 cycles

📝 Citation
If you use this code or findings in your research, please cite:

@article{author2026reassessing,
  title={Reassessing Battery SOH Prediction: Autoregressive Bias in Reported Performance},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2026}
}

📜 License
This project is licensed under the MIT License.

📧 Contact
For questions or suggestions, please open an issue on GitHub or contact:
[husnu.ulucay@saglik.gov.tr]

