Battery Health Prediction using CNN-TCN-LSTM with Attention

A deep-learning project for Lithium-Ion Battery State of Health (SOH)
estimation using NASA battery aging data. The project implements a
hybrid CNN-TCN-LSTM architecture with additive attention, an LSTM
baseline, leave-one-battery-out cross-validation, ablation experiments,
and visualization.

Pipeline

NASA Battery Data
       ↓
Data Loading
       ↓
Signal Smoothing
       ↓
Voltage-Domain Interpolation
       ↓
Electrochemical Feature Extraction
       ↓
Sequence Construction
       ↓
CNN → TCN → LSTM → Attention
       ↓
SOH Prediction
       ↓
Evaluation & Visualization

Dataset

Experiments use four NASA battery cells:

B0005

B0006

B0007

B0018

The processed sequence dataset has shape:

X: (samples, 20, 3, 300)
y: (samples,)

The three extracted feature channels are:

dQ/dV

dV/dQ

dI/dV

Preprocessing includes Savitzky-Golay smoothing, voltage-domain
interpolation from 2.5 V to 4.2 V using 300 points, numerical
derivatives, and temporal sequence construction.

Model

The main architecture is:

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

The CNN extracts local features from each cycle. The TCN uses dilated
convolutions with dilations 1, 2, 4 and 8 to model temporal patterns.
The LSTM models sequential dependencies, while additive attention
assigns importance to different sequence steps.

Training

Main configuration:

Optimizer: Adam
Learning rate: 0.001
Loss: MSE
Batch size: 64
Maximum epochs: 200
Early stopping patience: 20
Gradient clipping: 5.0

PyTorch uses CUDA when available and otherwise runs on CPU.

Leave-One-Battery-Out Cross-Validation

One complete battery is held out for testing while the remaining
batteries are used for training.

Battery                  MAE           RMSE              R²

B0005               0.097381       0.125371       -0.737823
B0006               0.094473       0.109599       -0.089520
B0007               0.061850       0.079056       -0.041903
B0018               0.092557       0.101428       -1.315784
Average     0.086565   0.103863   -0.546257

The negative average R² shows that the current hybrid model has limited
cross-battery generalization and still requires improvement.

LSTM Baseline

Battery                  MAE           RMSE              R²

B0005               0.080447       0.099260       -0.089329
B0006               0.125011       0.149667       -1.031768
B0007               0.045598       0.058007        0.439062
B0018               0.082472       0.100507       -1.273923
Average     0.083382   0.101860   -0.488990

The LSTM baseline slightly outperforms the full hybrid model on the
average metrics in the current experiment.

Ablation Study

Model                     MAE           RMSE             R²

CNN Only             0.076197       0.091754      -0.157223
CNN-LSTM     0.054995   0.068928   0.258955
CNN-TCN-LSTM         0.083947       0.095680      -0.399187
Full Model           0.080828       0.097948      -0.519467

The CNN-LSTM variant performed best in the current ablation experiment.
This suggests that adding TCN and attention did not improve
cross-battery generalization under the current experimental setup.

Visualizations

Generated results are stored under results/plots/, including:

Actual vs predicted SOH

Prediction error

Training history

Attention weights

Battery-wise predictions

Cross-validation predictions

Cross-validation errors and R²

Ablation results

Model comparisons

Repository Structure

battery-health-cnn-tcn-lstm/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── experiments/
│   ├── final_summary/
│   └── plots/
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
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── visualization/
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone https://github.com/GitankRana/battery-health-cnn-tcn-lstm.git
cd battery-health-cnn-tcn-lstm

Create a virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Running the Project

Build the processed data:

python scripts/build_dataset.py
python scripts/build_sequences.py

Train:

python scripts/train.py

Evaluate:

python scripts/evaluate.py

Run leave-one-battery-out cross-validation:

python scripts/cross_validate.py

Run the LSTM baseline:

python scripts/baseline_lstm.py

Run the ablation study:

python scripts/ablation_study.py

Generate analysis plots:

python scripts/generate_plots.py
python scripts/attention_plot.py
python scripts/battery_prediction_plots.py
python scripts/cv_prediction_plots.py
python scripts/model_comparison.py

Current Findings

The experiments demonstrate that model complexity does not automatically
improve cross-battery generalization. In the current experiments, the
simpler CNN-LSTM variant performed better than the full
CNN-TCN-LSTM-Attention model.

Potential future improvements include:

Better normalization across batteries

More robust validation strategies

Hyperparameter tuning

Feature selection

Sequence-length analysis

Additional regularization

Investigation of battery-specific degradation behavior

Additional baseline models

Reference

This project is intended as a reproduction/implementation based on:

Deep learning-based battery health prediction for enhancing electric
vehicle performance

Dataset: NASA battery aging data.

Author

Gitank Rana

GitHub: https://github.com/GitankRana
