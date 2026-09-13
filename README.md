🔋 Battery Health Prediction using CNN–TCN–LSTM with Attention

A deep-learning framework for Lithium-Ion Battery State of Health (SOH) estimation using NASA battery aging data.

This project investigates whether combining CNN, Temporal Convolutional Networks (TCN), LSTM, and Attention can improve battery health prediction and, more importantly, generalize to batteries that were never seen during training.

✨ Highlights
🧠 Hybrid CNN → TCN → LSTM → Attention architecture
🔋 NASA Lithium-Ion battery aging dataset
📈 Electrochemical feature extraction
🔄 Voltage-domain interpolation
🧹 Savitzky–Golay signal smoothing
⏱️ Temporal sequence modeling
🎯 Leave-One-Battery-Out cross-validation
🧪 LSTM baseline comparison
🔬 CNN / CNN-LSTM / CNN-TCN-LSTM ablation study
👀 Attention-weight visualization
📊 Battery-wise prediction and error analysis
🧩 Project Pipeline
NASA Battery Aging Data
          │
          ▼
     Data Loading
          │
          ▼
   Signal Preprocessing
   ├── Signal Smoothing
   ├── Voltage Interpolation
   └── Numerical Derivatives
          │
          ▼
  Electrochemical Features
   ├── dQ/dV
   ├── dV/dQ
   └── dI/dV
          │
          ▼
   Sequence Construction
      20 consecutive cycles
          │
          ▼
       CNN Feature
       Extraction
          │
          ▼
        TCN
   Temporal Patterns
          │
          ▼
        LSTM
 Sequential Dependencies
          │
          ▼
      Attention
 Important Cycle Weighting
          │
          ▼
    SOH Prediction
          │
          ▼
 Evaluation + Analysis
🔋 Dataset

The experiments use four NASA battery cells:

Battery	Role
B0005	Cross-validation fold
B0006	Cross-validation fold
B0007	Cross-validation fold
B0018	Cross-validation fold

The processed dataset contains:

X: (samples, 20, 3, 300)
y: (samples,)

Each sample therefore represents 20 consecutive battery cycles, with 3 feature channels, each sampled at 300 voltage points.

Extracted Features
Feature	Description
dQ/dV	Differential capacity with respect to voltage
dV/dQ	Differential voltage with respect to capacity
dI/dV	Differential current with respect to voltage
⚙️ Preprocessing

The raw battery discharge signals are transformed before being passed to the neural network.

1. Signal Smoothing

Savitzky–Golay filtering is applied to reduce noise while preserving the shape of the signal.

2. Voltage-Domain Interpolation

Signals are interpolated onto a common voltage grid:

Voltage range: 2.5 V → 4.2 V
Number of points: 300

This ensures that cycles with different numbers of raw measurements can be represented using the same input dimensions.

3. Electrochemical Feature Extraction

Numerical derivatives are calculated to obtain:

dQ/dV
dV/dQ
dI/dV
4. Temporal Sequence Construction

Twenty consecutive cycles are grouped into a single input sequence:

20 cycles × 3 features × 300 voltage points
🧠 Model Architecture

The main model follows:

Input
  ↓
CNN
  ↓
TCN
  ↓
LSTM
  ↓
Additive Attention
  ↓
Fully Connected Regressor
  ↓
SOH
CNN

The CNN extracts local patterns from the electrochemical feature curves of each battery cycle.

TCN

The Temporal Convolutional Network models temporal patterns using dilated convolutions.

Dilations:

1 → 2 → 4 → 8

This allows the network to capture patterns over different temporal scales.

LSTM

The LSTM processes the resulting sequence and learns sequential dependencies between battery cycles.

Attention

Additive attention assigns different importance to the 20 cycles in the input sequence.

This also allows the project to visualize which parts of the temporal sequence the model relies on most heavily.

🏋️ Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Loss	MSE
Batch Size	64
Maximum Epochs	200
Early Stopping Patience	20
Gradient Clipping	5.0
Device	CUDA if available, otherwise CPU
🧪 Leave-One-Battery-Out Cross-Validation

To evaluate cross-battery generalization, one entire battery is held out during each experiment.

For example:

Train:
B0006 + B0007 + B0018

Test:
B0005

The process is repeated for every battery.

Hybrid Model Results
Battery	MAE	RMSE	R²
B0005	0.097381	0.125371	-0.737823
B0006	0.094473	0.109599	-0.089520
B0007	0.061850	0.079056	-0.041903
B0018	0.092557	0.101428	-1.315784
Average	0.086565	0.103863	-0.546257
Key Observation

The negative average R² indicates that the current hybrid architecture does not generalize strongly across unseen batteries.

This is an important result rather than something to hide: the experiments show that increasing architectural complexity does not automatically produce better cross-battery performance.

📏 LSTM Baseline

A simpler LSTM model was implemented as a baseline for comparison.

Battery	MAE	RMSE	R²
B0005	0.080447	0.099260	-0.089329
B0006	0.125011	0.149667	-1.031768
B0007	0.045598	0.058007	0.439062
B0018	0.082472	0.100507	-1.273923
Average	0.083382	0.101860	-0.488990

The LSTM baseline slightly outperforms the full hybrid model on the average MAE, RMSE, and R² in the current experiment.

🔬 Ablation Study

To understand the contribution of individual architectural components, multiple variants were evaluated.

Model	MAE	RMSE	R²
CNN Only	0.076197	0.091754	-0.157223
CNN-LSTM	0.054995	0.068928	0.258955
CNN-TCN-LSTM	0.083947	0.095680	-0.399187
Full Model	0.080828	0.097948	-0.519467
🏆 Best Ablation

The CNN-LSTM configuration achieved the strongest results:

MAE  = 0.054995
RMSE = 0.068928
R²   = 0.258955

This suggests that, under the current experimental setup, adding the TCN and attention mechanisms did not improve generalization over the simpler CNN-LSTM architecture.

📊 Visualizations

The project generates a range of plots for model analysis.

Prediction Analysis
Actual vs Predicted SOH
Prediction Error
Battery-wise Predictions
Cross-validation Predictions
Training Analysis
Training / Validation History
Cross-validation Error
Cross-validation R²
Model Analysis
Attention Weights
Ablation Comparison
Model Comparison

All generated visualizations are stored under:

results/plots/
📁 Repository Structure
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
🚀 Installation
Clone the repository
git clone https://github.com/GitankRana/battery-health-cnn-tcn-lstm.git
cd battery-health-cnn-tcn-lstm
Create a virtual environment

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
Install dependencies
pip install -r requirements.txt
▶️ Running the Project
Build the dataset
python scripts/build_dataset.py
python scripts/build_sequences.py
Train the hybrid model
python scripts/train.py
Evaluate the model
python scripts/evaluate.py
Run cross-validation
python scripts/cross_validate.py
Run LSTM baseline
python scripts/baseline_lstm.py
Run ablation study
python scripts/ablation_study.py
Generate visualizations
python scripts/generate_plots.py
python scripts/attention_plot.py
python scripts/battery_prediction_plots.py
python scripts/cv_prediction_plots.py
python scripts/model_comparison.py
💡 Key Findings

The experiments reveal an important result:

More complex does not necessarily mean better.

Although the proposed architecture combines:

CNN + TCN + LSTM + Attention

the simpler CNN-LSTM model achieved better performance in the ablation study.

The current results also indicate that cross-battery generalization remains the primary challenge. Performance varies substantially between batteries, with several experiments producing negative R² values.

Potential directions for improvement include:

Better normalization across batteries
Hyperparameter optimization
Improved feature selection
Sequence-length experiments
Additional regularization
Battery-specific degradation analysis
More robust validation strategies
Additional baseline architectures
📚 Reference

Dataset: NASA Battery Aging Dataset

Project focus: Deep-learning-based Lithium-Ion battery State of Health estimation and cross-battery generalization.

👨‍💻 Author

Gitank Rana

Electrical Engineering
Netaji Subhas University of Technology (NSUT)

GitHub

⭐ Project Summary
Raw NASA Data
      ↓
Preprocessing
      ↓
Electrochemical Features
      ↓
20-Cycle Sequences
      ↓
CNN → TCN → LSTM → Attention
      ↓
SOH Prediction
      ↓
Cross-Validation
      ↓
Baseline + Ablation
      ↓
Analysis & Visualization

Goal: Build and evaluate deep-learning approaches for battery SOH prediction while explicitly testing how well the models generalize to previously unseen batteries.
