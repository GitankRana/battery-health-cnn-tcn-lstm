🔋 Battery Health Prediction with CNN–TCN–LSTM + Attention

A deep learning framework for estimating State of Health (SOH) in Lithium-Ion batteries, built on the NASA Battery Aging Dataset. The core question this project investigates isn't just "can we predict SOH?" — it's "can a model trained on some batteries generalize to a battery it has never seen?"

That second question turns out to be the hard one, and the honest answer here is: not yet, not with this architecture. This README documents the pipeline, the architecture, and — just as importantly — what the results actually show.

Table of Contents
Overview
Pipeline
Dataset
Preprocessing
Model Architecture
Training Configuration
Results
Leave-One-Battery-Out Cross-Validation
LSTM Baseline
Ablation Study
Key Findings
Visualizations
Repository Structure
Installation
Usage
Future Work
Reference
Author
Overview
	
🧠 Architecture	CNN → TCN → LSTM → Attention
🔋 Dataset	NASA Li-Ion Battery Aging Dataset
📈 Features	Electrochemical (dQ/dV, dV/dQ, dI/dV)
🔄 Preprocessing	Voltage-domain interpolation + Savitzky–Golay smoothing
🎯 Validation	Leave-One-Battery-Out cross-validation
🧪 Comparisons	LSTM baseline + component-wise ablation study
👀 Interpretability	Attention-weight visualization over cycles
Pipeline
NASA Battery Aging Data
        │
        ▼
   Data Loading
        │
        ▼
Signal Preprocessing
   ├── Smoothing (Savitzky–Golay)
   ├── Voltage-domain interpolation
   └── Numerical derivatives
        │
        ▼
Electrochemical Features
   ├── dQ/dV
   ├── dV/dQ
   └── dI/dV
        │
        ▼
Sequence Construction (20 consecutive cycles)
        │
        ▼
   CNN → TCN → LSTM → Attention
        │
        ▼
     SOH Prediction
        │
        ▼
 Evaluation & Analysis
Dataset

Four NASA battery cells are used, each serving as a cross-validation fold:

Battery	Role
B0005	CV fold
B0006	CV fold
B0007	CV fold
B0018	CV fold

Processed tensor shapes:

X: (samples, 20, 3, 300)   # 20 cycles × 3 feature channels × 300 voltage points
y: (samples,)              # SOH labels

Extracted features:

Feature	Description
dQ/dV	Differential capacity w.r.t. voltage
dV/dQ	Differential voltage w.r.t. capacity
dI/dV	Differential current w.r.t. voltage
Preprocessing
Signal smoothing — Savitzky–Golay filtering reduces noise while preserving curve shape.
Voltage-domain interpolation — all cycles are resampled onto a common grid (2.5 V → 4.2 V, 300 points), so cycles with different raw sample counts become directly comparable.
Electrochemical feature extraction — numerical derivatives yield dQ/dV, dV/dQ, and dI/dV.
Temporal sequence construction — 20 consecutive cycles are grouped into a single input of shape 20 × 3 × 300.
Model Architecture
Input
  │
  ▼
CNN            → extracts local patterns from each cycle's electrochemical curves
  │
  ▼
TCN            → dilated convolutions (dilations 1 → 2 → 4 → 8) capture multi-scale temporal patterns
  │
  ▼
LSTM           → models sequential dependencies across the 20 cycles
  │
  ▼
Additive Attention → learns which cycles matter most, and gives an interpretable weight per cycle
  │
  ▼
Fully Connected Regressor
  │
  ▼
SOH
Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Loss	MSE
Batch Size	64
Max Epochs	200
Early Stopping Patience	20
Gradient Clipping	5.0
Device	CUDA if available, else CPU
Results
Leave-One-Battery-Out Cross-Validation

Each experiment trains on three batteries and tests on the fourth, fully unseen, battery.

Full Hybrid Model (CNN–TCN–LSTM–Attention):

Battery	MAE	RMSE	R²
B0005	0.097381	0.125371	-0.737823
B0006	0.094473	0.109599	-0.089520
B0007	0.061850	0.079056	-0.041903
B0018	0.092557	0.101428	-1.315784
Average	0.086565	0.103863	-0.546257

The negative average R² means the model performs worse than simply predicting the mean SOH for unseen batteries. That's a real and important result, not a bug to paper over — it shows that this architecture, as configured, does not generalize well across batteries with different degradation behavior.

LSTM Baseline

A simpler single-branch LSTM was trained under the same protocol for comparison:

Battery	MAE	RMSE	R²
B0005	0.080447	0.099260	-0.089329
B0006	0.125011	0.149667	-1.031768
B0007	0.045598	0.058007	0.439062
B0018	0.082472	0.100507	-1.273923
Average	0.083382	0.101860	-0.488990

The plain LSTM slightly edges out the full hybrid model on every averaged metric.

Ablation Study
Model	MAE	RMSE	R²
CNN Only	0.076197	0.091754	-0.157223
CNN-LSTM	0.054995	0.068928	0.258955
CNN-TCN-LSTM	0.083947	0.095680	-0.399187
Full Model (+ Attention)	0.080828	0.097948	-0.519467

Best performer: CNN-LSTM, by a clear margin on every metric. Adding TCN and Attention on top made results worse under this experimental setup, not better.

Key Findings
More complexity ≠ better generalization. The simplest working combination (CNN-LSTM) outperformed the full CNN-TCN-LSTM-Attention stack across the board.
Cross-battery generalization is the real bottleneck. Performance is inconsistent across batteries, and several folds produce negative R², meaning the model hasn't learned degradation patterns that transfer between cells.
This is reported as a finding, not hidden. The value of the experiment is in showing where the architecture breaks down, which points directly at next steps (see below) rather than at scaling the model further.
Visualizations

Generated automatically and saved to results/plots/:

Prediction analysis

Actual vs. predicted SOH
Prediction error
Battery-wise predictions
Cross-validation predictions

Training analysis

Training / validation loss history
Cross-validation error
Cross-validation R²

Model analysis

Attention weight heatmaps
Ablation comparison
Model comparison (hybrid vs. baseline vs. ablations)
Repository Structure
battery-health-cnn-tcn-lstm/
│
├── data/
│   ├── raw/
│   │   └── NASA/
│   └── processed/
│
├── results/
│   ├── experiments/
│   ├── final_summary/
│   └── plots/
│
├── scripts/
│   ├── build_dataset.py
│   ├── build_sequences.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_validate.py
│   ├── baseline_lstm.py
│   ├── ablation_study.py
│   ├── model_comparison.py
│   ├── attention_plot.py
│   ├── battery_prediction_plots.py
│   ├── cv_prediction_plots.py
│   ├── generate_plots.py
│   └── final_summary.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── visualization/
│
├── requirements.txt
└── README.md
Installation

Clone the repository

bash
git clone https://github.com/GitankRana/battery-health-cnn-tcn-lstm.git
cd battery-health-cnn-tcn-lstm

Create a virtual environment (Windows PowerShell)

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies

bash
pip install -r requirements.txt
Usage
bash
# 1. Build the dataset
python scripts/build_dataset.py
python scripts/build_sequences.py

# 2. Train the hybrid model
python scripts/train.py

# 3. Evaluate
python scripts/evaluate.py

# 4. Leave-one-battery-out cross-validation
python scripts/cross_validate.py

# 5. LSTM baseline
python scripts/baseline_lstm.py

# 6. Ablation study
python scripts/ablation_study.py

# 7. Generate all visualizations
python scripts/generate_plots.py
python scripts/attention_plot.py
python scripts/battery_prediction_plots.py
python scripts/cv_prediction_plots.py
python scripts/model_comparison.py
Future Work

Cross-battery generalization is the main open problem. Promising directions:

Per-battery / global normalization strategies to reduce distribution shift between cells
Systematic hyperparameter optimization (currently fixed, not tuned)
Alternative or additional electrochemical features
Sequence-length experiments (is 20 cycles the right window?)
Stronger regularization (dropout, weight decay) to fight overfitting to training batteries
Battery-specific degradation curve analysis to understand why certain cells transfer poorly
Larger and more diverse validation sets across battery chemistries
Additional baseline architectures (Transformers, GRU, pure TCN) for a fuller comparison
Reference
Dataset: NASA Prognostics Center of Excellence — Battery Data Set
Focus: Deep-learning-based Lithium-Ion battery State of Health estimation, with an emphasis on honest evaluation of cross-battery generalization.
Author

Gitank Rana Electrical Engineering, Netaji Subhas University of Technology (NSUT) GitHub
