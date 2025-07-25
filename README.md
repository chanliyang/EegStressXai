# EegStressXai
This project seeks to use XAI to test results on different ML models to see what are the features that contribute to stress vs relax classification

---

# EEG-Based Stress Detection with Explainable AI

This research project implements a multi-step approach to detect stress using EEG data by leveraging machine learning models for classification and Explainable Artificial Intelligence (XAI) techniques for interpretability. The methodology is divided into stages: Data Acquisition and Preprocessing, Feature Extraction, Feature Selection, Machine Learning Model Training, and XAI Integration. The project is developed using Python in the Spyder Integrated Development Environment (IDE).

## Overview
<img width="944" height="352" alt="image" src="https://github.com/user-attachments/assets/bd823de0-5bb8-4483-8a3c-b185c466a3a1" />

The system aims to classify stress and non-stress states from EEG data using six machine learning models (SVM, KNN, Random Forest, Logistic Regression, Naive Bayes, and XGBoost) and enhance model transparency with SHAP and LIME. The workflow is summarized in the flowchart, which outlines the process and methods used in each stage.

## Methodology

### Data Acquisition and Preprocessing
- **Dataset**: The project utilizes the SAM40 dataset, an open-source EEG recording of 40 subjects performing cognitive tasks (e.g., Stroop test, arithmetic problem-solving) and relaxation phases to elicit short-term stress. Data was collected using a 32-channel Emotiv Epoc Flex gel kit at 128 Hz, segmented into 25-second epochs, and provided in MATLAB (.mat) format (raw and preprocessed with Savitzky-Golay filter and wavelet thresholding).
- **Preprocessing**: The clean dataset is used to focus on XAI effectiveness rather than preprocessing techniques. Data is split into training (trials 1 and 2, 320 files) and testing (trial 3, 160 files) sets (Code Segment 1). Epoching creates ~12 new 2-second samples per file to increase sample size.

<img width="688" height="362" alt="image" src="https://github.com/user-attachments/assets/a5439c76-9bc3-452f-bc3f-995f676264ec" />


### Feature Extraction
- **Purpose**: Condenses EEG signals into 16 meaningful features (Table 2) from time-domain (e.g., mean, Hjorth complexity) and frequency-domain (e.g., theta/alpha ratio, spectral entropy) to reduce dimensionality, prevent overfitting, and enable explainability.
- **Output**: 512 features (32 channels × 16 features) are extracted for both training and testing sets.

### Feature Selection
- **Method**: Principal Component Analysis (PCA) reduces 512 features to 25 components, optimizing variance retention and preventing overfitting. StandardScaler standardizes features (mean = 0, variance = 1) to improve model convergence.

### Machine Learning Model Training
- **Models**: Six models are trained on PCA-transformed features:
  - **SVM**: Uses kernel functions (rbf, linear) and GridSearchCV for hyperparameter tuning (C, gamma) with class_weight='balanced' 
  - **Random Forest**: Ensemble of decision trees with majority voting, tuned via n_estimators, max_depth, etc.
  - **KNN**: Classifies based on k-nearest neighbors (k=3,5,7,9) with Euclidean distance
  - **Logistic Regression**: Binary classifier with log-loss optimization, tuned with C and solvers
  - **Naive Bayes**: Probabilistic classifier with Gaussian assumption, tuned via var_smoothing
  - **XGBoost**: Gradient boosting with sequential decision trees, tuned with n_estimators, max_depth, and scale_pos_weight
- **Parameter Tuning**: GridSearchCV with 5-fold cross-validation optimizes hyperparameters based on F1-score due to class imbalance 

### Explainable Artificial Intelligence (XAI)
- **SHAP**: Uses KernelExplainer to compute Shapley values for global feature importance on 50 test samples, visualized via summary plots (Code Segment 9, Table 3).
- **LIME**: Applies LimeTabularExplainer for local explanations of single test instances, identifying top 10 PCA components with a bar plot (Code Segment 10).

### Performance Metrics
- **Metrics**: Evaluated using confusion matrix, accuracy, precision, recall, and F1-score. Bar plots visualize metrics except confusion matrix.
- **Focus**: F1-score is prioritized due to class imbalance (more "stress" than "not stress" data).

## Tools and Environment
The pipeline is implemented on a Windows environment using:
- **Python (3.x)**: Core language ([https://www.python.org/](https://www.python.org/))
- **NumPy**: Numerical computations ([https://numpy.org/](https://numpy.org/))
- **SciPy**: MATLAB data loading and PSD computation ([https://scipy.org/](https://scipy.org/))
- **scikit-learn**: Preprocessing, modeling, and metrics ([https://scikit-learn.org/stable/](https://scikit-learn.org/stable/))
- **XGBoost**: Gradient boosting ([https://xgboost.readthedocs.io/en/stable/](https://xgboost.readthedocs.io/en/stable/))
- **Matplotlib**: Visualizations ([https://matplotlib.org/](https://matplotlib.org/))
- **Seaborn**: Enhanced plots ([https://seaborn.pydata.org/](https://seaborn.pydata.org/))
- **SHAP**: Global explainability ([https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/))
- **LIME**: Local explainability ([https://lime-ml.readthedocs.io/en/latest/](https://lime-ml.readthedocs.io/en/latest/))
- **MNE**: EEG signal processing ([https://mne.tools/stable/index.html](https://mne.tools/stable/index.html))
- **IDE**: Spyder

## Getting Started
1. **Clone the Repository**: Download or clone this project to your local machine.
2. **Install Dependencies**: Ensure Python 3.x and the listed libraries are installed (use `pip install -r requirements.txt` if a requirements file is provided).
3. **Dataset**: Download the SAM40 dataset (link under Acknowledgement section) and place the `.mat` files in the specified local directory.
4. **Run the Code**: Open Spyder, load the main script, and execute to process data, train models, and generate results.


## Acknowledgments
- SAM40 dataset providers (https://www.kaggle.com/datasets/ayushtibrewal/raw-eeg-stress-dataset-sam40)
- scikit-learn, SHAP, LIME, and other library developers for open-source tools.

---

### Notes:
- **Structure**: The README is organized with headings and bullet points for readability, aligning with GitHub Markdown conventions.
