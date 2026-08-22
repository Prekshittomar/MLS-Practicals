# Machine Learning Safety - CARLA Model Safety Case

This repository contains the practical assignments and experiments used to generate the evidence for the CARLA miniature safety case. It evaluates perception models across key machine learning safety dimensions: In-distribution Accuracy, Adversarial Robustness, Calibration, Out-of-Distribution (OOD) Detection, and Explainability.

## Repository Structure & Reproducibility

The repository is organized into 6 practical folders. Each folder contains a Jupyter Notebook (`.ipynb`) that implements and evaluates specific safety verification steps. 

To reproduce the results for the safety case, run the notebooks in the following order:

### 1. Training, Testing, and In-Distribution Accuracy (V-1)
**Location:** `Practical - Training & Testing the CARLA model/ex_3_and_4.ipynb`
- **Purpose:** Trains the ResNet18 models for the pedestrian, traffic light, and vehicle classes (or evaluates pre-trained weights) and calculates the baseline performance metrics.
- **How to Reproduce:** Open the notebook, ensure your dataset paths are correctly set, and run all cells. The notebook will output a dataframe (`results_df`) displaying the **Accuracy, Precision, Recall, and F1-score** for each class on the in-distribution test set. Use the `Recall` metric to fill the V-1 evidence table.

### 2. Adversarial Robustness (V-2)
**Location:** `Practical - Attacking the CARLA Model/ex_8.ipynb`
- **Purpose:** Evaluates the models' robustness against small input perturbations using the Fast Gradient Sign Method (FGSM) attack.
- **How to Reproduce:** Run the cells to perform the FGSM attack on the test images. The notebook will output the **Clean Recall**, **Adversarial Recall**, and the **Recall Drop** for different $\varepsilon$ thresholds (e.g., $\varepsilon = 0.05$). Use the metrics at $\varepsilon = 0.05$ to fill the V-2 evidence table.

### 3. Calibrated Uncertainty (V-3)
**Location:** `Practical - Calibration in the CARLA Model/Calibration in the CARLA Model.ipynb`
- **Purpose:** Calibrates the models' confidence estimates using Temperature Scaling and visualizes reliability diagrams.
- **How to Reproduce:** Execute the notebook to extract logits from the validation set, optimize the Temperature ($T$) using Negative Log-Likelihood, and evaluate on the test set. It will print the **Expected Calibration Error (ECE)** before ($T=1.0$) and after scaling. Use these values to fill the V-3 evidence table.

### 4. Out-of-Distribution (OOD) Detection (V-4)
**Location:** `Practical - OOD Detection for the CARLA Model/ex_7.ipynb`
- **Purpose:** Evaluates anomaly detectors to flag inputs that fall outside the training distribution (e.g., fog or night conditions).
- **How to Reproduce:** Run the cells to extract feature representations and calculate confidence scores for the in-distribution and OOD test sets. The notebook evaluates both baseline Maximum Softmax Probability (MSP) and feature-based $k$-NN detectors, outputting the **AUROC** scores. Use the AUROC results to fill the V-4 evidence table.

### 5. Explainability 
**Location:** `Practical - Explaining the CARLA Model/6_5.ipynb` and `6_6.ipynb`
- **Purpose:** Provides interpretability for the models' predictions using saliency maps and feature attribution methods to ensure the models aren't relying on background shortcuts.
- **How to Reproduce:** Run the notebooks to generate visual explanations for specific test images.

### 6. Backdoor Attacks
**Location:** `Practical -  Calibration & Backdoor Attacks on the CARLA Model/ex_5.ipynb`
- **Purpose:** Explores the vulnerability of the CARLA models to targeted data poisoning and backdoor triggers.
- **How to Reproduce:** Execute the cells to evaluate the attack success rate (ASR) and the impact of the backdoor on clean recall.

## Prerequisites
To execute these notebooks, you will need:
- Python 3.8+
- PyTorch & Torchvision
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

Ensure your data directories (`train`, `test`, `test-fog`, `test-night`, `test-town-01`) and `models` folder are properly linked within the notebooks before execution.
