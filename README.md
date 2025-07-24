# ECE361 Final Project: Signal Detection with Statistical Modeling

This project applies statistical and signal processing methods to distinguish between "Target Absent" and "Target Present" signals in noisy environments using MATLAB and statistical tools.

## 📌 Objectives
- Identify optimal detection thresholds using ROC curves and Youden's Index.
- Fit Rayleigh and Rician distributions to experimental data.
- Apply bootstrapping for parameter estimation.
- Evaluate and compare multiple decision strategies (Arithmetic Mean, Geometric Mean, Max).

## 🔬 Key Results
- Best-fit distributions:
  - **Rayleigh** for Target Absent (H₀)
  - **Rician** for Target Present (H₁)
- Best thresholding method: **Geometric Mean** (AUC: 0.941, Error Rate: 13/130)
- Demonstrated the value of preprocessing in enhancing classification performance.

## 🛠 Tools Used
- MATLAB
- Bootstrapping techniques
- ROC/AUC analysis

## 📁 Contents
- `Part1_Threshold_Analysis.pdf`: ROC, confusion matrix, initial thresholds.
- `Part2_Distribution_Fitting.pdf`: Chi-squared tests, bootstrapping.
- `Part3_Complete_Project_Report.pdf`: Final modeling, comparison of methods.
