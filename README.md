# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-4DABCF?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-8CAAE6)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-2E8B57)

**Author:** MUKUNZI NDAHIRO James
**Course:** Machine Learning Technique 1
**Assessment:** Formative Assignment 1

---

## 1. Overview

This project is a research-informed empirical investigation into **one-step-ahead mobile network traffic forecasting** using the Milan telecommunications activity dataset (Barlacchi et al., 2015). It addresses the research question:

> *How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?*

The study is presented as a small empirical research project covering efficient large-scale data handling, exploratory and statistical time-series analysis, related-work-driven model selection, and a comparative forecasting experiment across the three highest-traffic areas of the city for the evaluation week of **16-22 December 2013**.

Three deliberately different sequential models are implemented and compared:

| Model | Family | Why it was selected |
|-------|--------|---------------------|
| **SARIMA** | Classical statistical | Captures linear autocorrelation and the strong daily cycle; interpretable benchmark |
| **LSTM** | Recurrent neural network | Learns nonlinear short- and long-range temporal dependencies from a sliding window |
| **1D-CNN** | Convolutional neural network | Extracts local temporal patterns; architecturally distinct nonlinear alternative |

A trivial **persistence baseline** (predict the next value equal to the current one) is included as a rigorous floor for the comparison.

## 2. Dataset

- **Source:** Milan grid, `sms-call-internet-mi-YYYY-MM-DD.txt` (Harvard Dataverse, DOI: 10.7910/DVN/EGZHFV).
- **Period:** 1 November 2013 to 1 January 2014 (62 daily files, ~20 GB of tab-separated text).
- **Structure:** the city is divided into a 10,000-cell (100x100) grid, sampled at 10-minute intervals.
- **Columns:** `square_id | timestamp(ms) | country_code | sms_in | sms_out | call_in | call_out | internet`.

Only the Internet-traffic signal is used; per-country records are aggregated by summing over country codes to obtain one value per `(square_id, timestamp)`.

> **Note:** the raw `data/` folder and generated `figures/` are excluded from version control because of their size. Instructions to regenerate them are below.

## 3. Repository Structure

```
.
├── mobile_network_traffic_forecasting.ipynb   # main analysis notebook (all work)
├── requirements.txt                           # pinned dependencies
├── README.md                                  # this file
├── .gitignore
├── data/
│   ├── raw/                                    # 62 daily .txt files (not tracked)
│   └── processed/                              # slim per-day Parquet files (generated)
└── figures/                                    # exported plots (generated)
```

## 4. Setup

Requires **Python 3.9+**. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Important:** `torch==2.2.2` requires `numpy<2`. The pinned `numpy==1.26.4` in `requirements.txt` satisfies this constraint; do not upgrade NumPy independently.

## 5. Data Preparation

1. Download the 62 daily files from the dataset source above.
2. Place all `.txt` files in `data/raw/`:

```
data/raw/sms-call-internet-mi-2013-11-01.txt
...
data/raw/sms-call-internet-mi-2014-01-01.txt
```

The notebook handles everything else: streaming conversion to Parquet, aggregation, analysis, and modelling.

## 6. Running the Project

Open and run the notebook top to bottom:

```bash
jupyter notebook mobile_network_traffic_forecasting.ipynb
```

Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  mobile_network_traffic_forecasting.ipynb
```

**Runtime notes**
- The one-off raw-to-Parquet conversion (~20 GB) runs once and is cached; subsequent loads take seconds.
- Hyperparameter grids and model training on CPU make a full end-to-end run take roughly **25-35 minutes**.
- Figures are written to `figures/` as they are produced.

## 7. Methodology Summary

- **Input representation:** a sliding window of the previous 144 ten-minute steps (one full day) predicts the next step.
- **Preprocessing:** per-series standardisation (zero mean, unit variance) fitted on the training split only, then applied to validation and test to avoid leakage; predictions are inverse-transformed before scoring.
- **Split:** chronological — training up to 1 Dec, validation 2-15 Dec, test = the evaluation week 16-22 Dec.
- **Tuning:** grid search (`sklearn.model_selection.ParameterGrid`) with validation-based selection, documented in the notebook.
- **Metrics:** MAE, RMSE and MAPE, reported per model and per area.

## 8. What the Notebook Produces

- **Data handling:** before/after memory evidence (~93% in-memory reduction; ~20 GB of text reduced to a few hundred MB of Parquet) with a discussion of trade-offs.
- **Exploratory analysis:** spatial distribution and 100x100 grid heatmap, top-3 area identification, two-week series for five areas.
- **Time-series analysis:** daily/weekly profiles, ACF/PACF, STL decomposition, and the ADF stationarity test.
- **Forecasting:** documented hyperparameter search, 9 actual-vs-predicted plots, three per-area metric tables, training/inference timing, and comparison against the persistence baseline.
- **Failure analysis:** error-by-time-of-day and peak-period breakdown with reasoning.

## 9. Key Results

- The **LSTM is the most accurate model** on every metric in every area and the **only model to beat the persistence baseline** (~9% mean MAE improvement).
- **SARIMA** reconstructs the daily cycle cleanly but sits around baseline accuracy at this short horizon.
- The **1D-CNN** is competitive only in smoother areas and degrades in the spikiest, highest-traffic cell.
- The model ranking is stable across areas, but the performance gap widens with local traffic burstiness.

## 10. Hardware

Timing statistics were recorded on macOS (Intel x86-64, 8 logical cores, 16 GB RAM), with PyTorch running on CPU. Exact values are reported in the notebook.

## 11. References

Full references are given in IEEE style in the final section of the notebook. Primary dataset reference:

G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, art. 150055, 2015, doi: 10.1038/sdata.2015.55.
