Battery Health Prediction with CNN–TCN–LSTM and Attention

A deep learning framework for estimating the State of Health (SOH) of lithium-ion batteries using NASA battery aging data.

This project investigates whether combining CNNs, Temporal Convolutional Networks (TCNs), LSTMs, and Attention can improve battery-health prediction across previously unseen battery cells. Alongside the proposed hybrid architecture, the project includes an LSTM baseline, leave-one-battery-out cross-validation, ablation experiments, and detailed visualization.

Overview

Accurate battery health estimation is important for applications such as electric vehicles, battery management systems, and energy storage.

The goal of this project is to learn degradation patterns from battery cycling data and predict the health of a battery from its electrochemical characteristics.

The overall pipeline is:

NASA Battery Data
↓
Signal Preprocessing
↓
Electrochemical Feature Extraction
↓
Sequence Construction
↓
CNN Feature Extraction
↓
TCN Temporal Modeling
↓
LSTM Sequential Modeling
↓
Attention Mechanism
↓
SOH Prediction

Dataset

The experiments use four battery cells from the NASA battery aging dataset:

B0005
B0006
B0007
B0018

The processed dataset contains sequences of 20 consecutive battery cycles, with three electrochemical feature channels evaluated over a 300-point voltage grid.

X shape: (samples, 20, 3, 300)
y shape: (samples,)
Extracted Features

The model uses:

dQ/dV — differential capacity with respect to voltage
dV/dQ — differential voltage with respect to capacity
dI/dV — differential current with respect to voltage

These features are extracted from the battery's voltage, current, and time measurements.

Preprocessing

The raw battery signals are processed before being provided to the neural network.

The preprocessing pipeline includes:

Signal cleaning and conversion to numerical arrays
Savitzky–Golay smoothing
Voltage-domain interpolation
Resampling onto a fixed voltage grid from 2.5 V to 4.2 V
Interpolation to 300 points
Numerical derivative calculation
Electrochemical feature extraction
Construction of temporal sequences from consecutive cycles

This converts variable-length battery-cycle measurements into a consistent representation that can be processed by the neural network.

Model Architecture

The main model is a:

CNN → TCN → LSTM → Attention → Regression

CNN

The CNN processes the electrochemical feature representation of each battery cycle.

It uses 1D convolutional layers to learn local patterns in the extracted electrochemical signals.

TCN

The Temporal Convolutional Network models temporal relationships between battery cycles.

The TCN uses dilated convolutions with dilation factors:

1 → 2 → 4 → 8

This allows the network to capture temporal patterns over different receptive-field sizes.

LSTM

The LSTM receives the temporal representations produced by the TCN and learns sequential dependencies associated with battery degradation.

Attention

An additive attention mechanism is applied to the LSTM outputs.

Rather than treating every cycle in the sequence equally, attention learns weights indicating which time steps contribute more strongly to the final prediction.

Regression Head

The resulting context vector is passed through fully connected layers and a sigmoid output layer to produce the predicted SOH.

Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Loss	MSE
Batch Size	64
Maximum Epochs	200
Early Stopping Patience	20
Gradient Clipping	5.0
Device	CUDA if available, otherwise CPU
Evaluation Strategy

A major focus of the project is cross-battery generalization.

Instead of randomly mixing samples from every battery, the experiments use Leave-One-Battery-Out Cross-Validation.

For each experiment:

One complete battery is held out for testing.
The remaining batteries are used for training.
The model therefore has to predict the degradation behavior of a battery it has not seen during training.

For example:

Train: B0005 + B0006 + B0007
Test:  B0018

This process is repeated for all four batteries.

Hybrid Model Results
Leave-One-Battery-Out Cross-Validation
Battery	MAE	RMSE	R²
B0005	0.097381	0.125371	-0.737823
B0006	0.094473	0.109599	-0.089520
B0007	0.061850	0.079056	-0.041903
B0018	0.092557	0.101428	-1.315784
Average	0.086565	0.103863	-0.546257

The results show that the current hybrid architecture has limited generalization across different battery cells. In particular, the negative average R² indicates that the model does not yet consistently outperform a simple mean-based prediction when evaluated on unseen batteries.

LSTM Baseline

To determine whether the additional architectural components actually provide an advantage, an LSTM baseline was trained using the same experimental framework.

Battery	MAE	RMSE	R²
B0005	0.080447	0.099260	-0.089329
B0006	0.125011	0.149667	-1.031768
B0007	0.045598	0.058007	0.439062
B0018	0.082472	0.100507	-1.273923
Average	0.083382	0.101860	-0.488990

The LSTM baseline slightly outperforms the full hybrid model on the average MAE and RMSE metrics in the current experiment.

This is an important result: increased model complexity does not necessarily translate into better cross-battery performance.

Ablation Study

An ablation study was performed to understand the contribution of different components of the architecture.

Model	MAE	RMSE	R²
CNN Only	0.076197	0.091754	-0.157223
CNN-LSTM	0.054995	0.068928	0.258955
CNN-TCN-LSTM	0.083947	0.095680	-0.399187
Full Model	0.080828	0.097948	-0.519467

The CNN-LSTM configuration produced the strongest results in the current ablation experiment.

Interestingly, adding the TCN and attention components did not improve performance under the current experimental setup.

This suggests that the degradation patterns represented in the dataset may not benefit from the additional temporal modeling complexity in its current form.

Results & Visualizations

The project generates a range of plots for analysing model behaviour and experimental results.

Prediction Analysis
Actual vs. predicted SOH
Battery-wise prediction curves
Prediction errors
Cross-validation predictions
Training Analysis
Training and validation loss
Learning behaviour during training
Model Analysis
Attention weights
Model-to-model comparison
MAE comparison
RMSE comparison
R² comparison
Ablation Analysis
CNN-only performance
CNN-LSTM performance
CNN-TCN-LSTM performance
Full CNN-TCN-LSTM-Attention performance

All generated visualizations are available under:

results/plots/
Project Structure
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
Getting Started
1. Clone the repository
git clone https://github.com/GitankRana/battery-health-cnn-tcn-lstm.git
cd battery-health-cnn-tcn-lstm
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
Running the Pipeline
Build the dataset
python scripts/build_dataset.py
python scripts/build_sequences.py
Train the hybrid model
python scripts/train.py
Evaluate the model
python scripts/evaluate.py
Run cross-validation
python scripts/cross_validate.py
Run the LSTM baseline
python scripts/baseline_lstm.py
Run the ablation study
python scripts/ablation_study.py
Generate visualizations
python scripts/generate_plots.py
python scripts/attention_plot.py
python scripts/battery_prediction_plots.py
python scripts/cv_prediction_plots.py
python scripts/model_comparison.py
Key Findings

The experiments lead to a useful conclusion:

More complex does not necessarily mean better.

The full CNN-TCN-LSTM-Attention architecture did not outperform the simpler models in the current cross-battery evaluation.

The CNN-LSTM ablation achieved the best overall performance among the evaluated architectures, reaching:

MAE  = 0.054995
RMSE = 0.068928
R²   = 0.258955

Meanwhile, the full hybrid model achieved:

MAE  = 0.080828
RMSE = 0.097948
R²   = -0.519467

This highlights the importance of model ablation and rigorous battery-level validation rather than evaluating a deep-learning architecture solely on training performance.

Future Work

Potential directions for improving the project include:

Better cross-battery normalization
Hyperparameter optimization
Improved feature selection
Investigation of different sequence lengths
Additional regularization strategies
Alternative temporal architectures
Battery-specific degradation analysis
Additional baseline models
More robust validation strategies
Reference

The project uses the NASA battery aging dataset and is intended as a deep-learning implementation for lithium-ion battery health prediction.

Author

Gitank Rana

GitHub: https://github.com/GitankRana
