# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:44:10 2025

@author: Raymo
"""
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.time_frequency import psd_array_welch
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import skew
import seaborn as sns
import warnings
# Suppress tight_layout warnings
warnings.filterwarnings("ignore", message="Tight layout not applied")

#---------------------------------------------Load Data-----------------------------------------------
def load_eeg_data(data_path, subjects=40, tasks=4, trials=3):
    print("\nStep 1: Loading EEG data")
    train_data, train_labels, test_data, test_labels = [], [], [], []
    task_names = ['Stroop', 'Arithmetic', 'Mirror_image', 'Relax']
    
    for subject in range(1, subjects + 1):
        for task_idx, task in enumerate(task_names):
            for trial in range(1, trials + 1):
                file_name = f"{task}_sub_{subject}_trial{trial}.mat"
                file_path = os.path.join(data_path, file_name)
                if os.path.exists(file_path):
                    try:
                        mat = scipy.io.loadmat(file_path)
                        eeg_data = mat['Clean_data']
                        if eeg_data.shape == (32, 3200):
                            label = 1 if task in ['Stroop', 'Arithmetic', 'Mirror_image'] else 0
                            if trial in [1, 2]:
                                train_data.append(eeg_data)
                                train_labels.append(label)
                            elif trial == 3:
                                test_data.append(eeg_data)
                                test_labels.append(label)
                    except Exception as e:
                        print(f"Warning: Failed to load {file_name}: {e}")
    print(f"Data loaded: {len(train_data)} training samples, {len(test_data)} test samples")
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

#---------------------------------------------Feature Extraction-----------------------------------------------
def extract_features(data, labels, sfreq=128, epoch_size=2, dataset_type="training"):
    print(f"\nStep 2: Extracting features with epoching ({dataset_type} set)")
    window_size = epoch_size * sfreq
    all_features, all_labels = [], []
    
    feature_names = []
    n_channels = 32
    for ch in range(n_channels):
        feature_names.extend([
            f"EEG{ch+1}_mean", f"EEG{ch+1}_median", f"EEG{ch+1}_std", f"EEG{ch+1}_peak_to_peak",
            f"EEG{ch+1}_min", f"EEG{ch+1}_max", f"EEG{ch+1}_hjorth_activity",
            f"EEG{ch+1}_hjorth_mobility", f"EEG{ch+1}_hjorth_complexity",
            f"EEG{ch+1}_theta_power", f"EEG{ch+1}_alpha_power", f"EEG{ch+1}_beta_power",
            f"EEG{ch+1}_gamma_power", f"EEG{ch+1}_theta_alpha_ratio",
            f"EEG{ch+1}_spectral_entropy", f"EEG{ch+1}_sample_entropy"
        ])
    
    print(f"Epoching with {epoch_size}-second windows")
    print(f"Features extracted: mean, median, std, peak-to-peak, min, max, Hjorth parameters, "
          f"theta/alpha/beta/gamma power, theta-alpha ratio, spectral entropy, sample entropy "
          f"for each of {n_channels} EEG channels")
    
    for trial_idx, trial in enumerate(data):
        n_epochs = trial.shape[-1] // window_size
        for i in range(n_epochs):
            window = trial[:, i*window_size:(i+1)*window_size]
            window_features = []
            
            # Time-domain features
            for ch in range(n_channels):
                data = window[ch, :]
                mean = np.mean(data)
                median = np.median(data)
                std = np.std(data)
                peak_to_peak = np.ptp(data)
                min_val = np.min(data)
                max_val = np.max(data)
                # Hjorth features
                activity = np.var(data)
                diff = np.diff(data)
                mobility = np.sqrt(np.var(diff) / activity if activity != 0 else 0)
                diff2 = np.diff(diff)
                complexity = (np.sqrt(np.var(diff2) / np.var(diff) if np.var(diff) != 0 else 0) /
                              (mobility if mobility != 0 else 1e-10))
                window_features.extend([mean, median, std, peak_to_peak, min_val, max_val,
                                        activity, mobility, complexity])
            
            # Frequency-domain features
            psds, freqs = psd_array_welch(window, sfreq=sfreq, fmin=0.5, fmax=44, n_fft=window_size, verbose=False)
            freq_bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 44)}
            for ch in range(n_channels):
                channel_psd = psds[ch, :]
                total_power = np.sum(channel_psd[(freqs >= 4) & (freqs <= 44)])
                if total_power == 0:
                    print(f"Warning: Zero total power in channel {ch+1}")
                for band, (f_low, f_high) in freq_bands.items():
                    band_idx = (freqs >= f_low) & (freqs <= f_high)
                    band_power = np.sum(channel_psd[band_idx]) / (total_power + 1e-10)
                    window_features.append(band_power)
                # Theta-alpha ratio
                theta_power = np.sum(channel_psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.sum(channel_psd[(freqs >= 8) & (freqs <= 13)])
                theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0
                window_features.append(theta_alpha_ratio)
                # Spectral entropy
                psd_norm = channel_psd / np.sum(channel_psd) if np.sum(channel_psd) > 0 else channel_psd
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10)) if np.any(psd_norm > 0) else 0
                window_features.append(spectral_entropy)
                # Sample entropy (using skew as placeholder)
                sample_entropy = skew(data)
                window_features.append(sample_entropy)
            
            all_features.append(window_features)
            all_labels.append(labels[trial_idx])
    
    print(f"Features extracted: {len(all_features)} samples with {len(feature_names)} features each ({dataset_type} set)")
    return np.array(all_features), np.array(all_labels), feature_names

#---------------------------------------------ML Models-----------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, n_components=25, pca_x=0, pca_y=1):
    print("\nStep 3: Applying PCA and scaling features")
    # PCA for model training
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Map PCA components to original features
    pca_loadings = pca.components_
    pca_feature_names = []
    print(f"PCA applied: Reduced to {n_components} components. Component mappings:")
    for i, component in enumerate(pca_loadings):
        top_features_idx = np.argsort(np.abs(component))[-3:]  # Top 3 contributing features
        top_features = [f"{feature_names[idx]} ({component[idx]:.4f})" for idx in top_features_idx]
        pca_feature_names.append(f"PCA_{i+1} ({', '.join(top_features)})")
        print(f"PCA_{i+1}: {', '.join(top_features)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    print("PCA and scaling completed")

    # --- Visualization 1: Correlation Heatmap of Original 512 Features ---
    print("\nVisualizing correlation of original 512 features (subset)")
    plt.figure(figsize=(12, 10))
    subset_features = X_train[:, :20]  # Adjust as needed
    subset_feature_names = feature_names[:20]
    corr_matrix = np.corrcoef(subset_features, rowvar=False)
    sns.heatmap(corr_matrix, xticklabels=subset_feature_names, yticklabels=subset_feature_names,
                cmap='coolwarm', vmin=-1, vmax=1, annot=False)
    plt.title("Correlation Heatmap of Original EEG Features (Subset)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Histogram of Top Features from PCA_1 ---
    print(f"\nVisualizing distribution of top features in PCA_{pca_x+1}")
    top_features_idx = np.argsort(np.abs(pca_loadings[pca_x]))[-3:]  # Top 3 features for pca_x
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(top_features_idx):
        plt.subplot(1, 3, i+1)
        sns.histplot(X_train[:, idx], kde=True, color='skyblue', label='Train')
        sns.histplot(X_test[:, idx], kde=True, color='orange', label='Test')
        plt.title(f"{feature_names[idx]}", fontsize=10)
        plt.xlabel("Value")
        plt.legend()
    plt.suptitle(f"Distribution of Top 3 Features for PCA_{pca_x+1}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- Visualization 3: Scatter Plot of PCA_x vs PCA_y ---
    print(f"\nVisualizing PCA_{pca_x+1} vs PCA_{pca_y+1} scatter plot")
    plt.figure(figsize=(10, 6))
    for label, color, marker in zip([0, 1], ['blue', 'red'], ['o', '^']):
        mask_train = y_train == label
        mask_test = y_test == label
        plt.scatter(X_train_scaled[mask_train, pca_x], X_train_scaled[mask_train, pca_y],
                    c=color, marker=marker, label=f"Train {['Not Stress', 'Stress'][label]}", alpha=0.5)
        plt.scatter(X_test_scaled[mask_test, pca_x], X_test_scaled[mask_test, pca_y],
                    c=color, marker=marker, label=f"Test {['Not Stress', 'Stress'][label]}", alpha=0.2)
    plt.xlabel(f"PCA_{pca_x+1}")
    plt.ylabel(f"PCA_{pca_y+1}")
    plt.title(f"PCA_{pca_x+1} vs PCA_{pca_y+1} (Scaled) for Training and Test Data", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visualization 4: Explained Variance Ratio ---
    print("\nVisualizing explained variance ratio of PCA components")
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components+1), explained_variance, marker='o', linestyle='--')
    plt.xlabel("PCA Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"Explained Variance Ratio of {n_components} PCA Components", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    models = {
        'SVM': {
            'model': SVC(random_state=42, class_weight='balanced', probability=True),
            'params': {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01, 0.1]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
         },
    }
    results, accuracies = {}, {}
    class_names = ['Not Stress', 'Stress']
    
    # --- Visualization 5: Model Decision Boundaries ---
    print(f"\nVisualizing model decision boundaries for PCA_{pca_x+1} vs PCA_{pca_y+1}")
    x_min, x_max = X_train_scaled[:, pca_x].min() - 1, X_train_scaled[:, pca_x].max() + 1
    y_min, y_max = X_train_scaled[:, pca_y].min() - 1, X_train_scaled[:, pca_y].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    for name, config in models.items():
        print(f"Training {name}")
        grid = GridSearchCV(config['model'], config['params'], cv=5, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        y_pred = grid.best_estimator_.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        results[name] = {
            'accuracy': accuracy,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_
        }
 
        # Performance Metrics
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print("Confusion Matrix:")
        print("\t\tPredicted Stress\tPredicted Not Stress")
        print(f"Actual Stress\t{cm[1,1]}\t\t{cm[1,0]}")
        print(f"Actual Not Stress\t{cm[0,1]}\t\t{cm[0,0]}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f} \n")
        
        # Plot decision boundary for PCA_x vs PCA_y
        plt.figure(figsize=(10, 6))
        Z = grid.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], n_components-2))])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
        for label, color, marker in zip([0, 1], ['blue', 'red'], ['o', '^']):
            mask_train = y_train == label
            plt.scatter(X_train_scaled[mask_train, pca_x], X_train_scaled[mask_train, pca_y],
                        c=color, marker=marker, label=f"Train {class_names[label]}", alpha=0.5)
        plt.xlabel(f"PCA_{pca_x+1}")
        plt.ylabel(f"PCA_{pca_y+1}")
        plt.title(f"Decision Boundary for {name} (PCA_{pca_x+1} vs PCA_{pca_y+1})", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Visualization 6: Feature Importance for Tree-Based Models ---
        if name in ['Random Forest', 'XGBoost']:
            print(f"Visualizing feature importance for {name}")
            importances = grid.best_estimator_.feature_importances_
            plt.figure(figsize=(12, 6))
            plt.bar(range(n_components), importances, tick_label=[f"PCA_{i+1}" for i in range(n_components)])
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("PCA Component")
            plt.ylabel("Feature Importance")
            plt.title(f"Feature Importance for {name}", fontsize=14)
            plt.tight_layout()
            plt.show()
            
        # Apply XAI to the current model using PCA-transformed data
        apply_xai_to_model(name, grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, pca, feature_names, class_names, n_components)
    
    # Plot accuracies
    print("Step 6: Plotting model accuracy comparison")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, pad=15)
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return results
#---------------------------------------------XAI-----------------------------------------------
def apply_xai_to_model(model_name, model, X_train, X_test, y_train, pca, feature_names, class_names, n_components=25):
    print(f"\nApplying XAI to {model_name}")
    
    # Map PCA components to original features with simplified names for SHAP
    pca_loadings = pca.components_
    pca_feature_names = [f"PCA_{i+1}" for i in range(n_components)]
    for i, component in enumerate(pca_loadings):
        top_features_idx = np.argsort(np.abs(component))[-3:]
        top_features = [f"{feature_names[idx]} ({component[idx]:.4f})" for idx in top_features_idx]
        pca_feature_names[i] = f"{pca_feature_names[i]} ({', '.join(top_features)})"
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # SHAP Explanation
    print(f"Generating SHAP summary plot for {model_name}")
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
    shap_values = explainer.shap_values(X_test[:50], nsamples=100, silent=True)
    
    # Ensure we get the SHAP values for the positive class ("Stress")
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[1]  # Positive class ("Stress")
    else:
        shap_values_to_plot = shap_values
    if hasattr(shap_values_to_plot, 'ndim') and shap_values_to_plot.ndim == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 1]  # Take the "Stress" class
    plt.figure(figsize=(20, 12)) 
    shap.summary_plot(shap_values_to_plot, X_test[:50], feature_names=pca_feature_names, 
                      class_names=class_names, show=False, plot_type="dot", max_display=7)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.title(f"SHAP Summary Plot for {model_name}", fontsize=14, pad=15)
    plt.xlabel("SHAP Value - Low Impact (Not Stress) to High Impact (Stress)")
    plt.tight_layout()
    plt.show()
    
    # LIME Explanation with simplified labels
    print(f"Generating LIME explanation for {model_name}")
    simplified_pca_names = [f"PCA_{i+1} ({X_test[0][i]:.4f})" for i in range(n_components)]
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=simplified_pca_names,
        class_names=class_names, mode='classification', random_state=42
    )
    lime_exp = lime_explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
    plt.figure(figsize=(10, 6))
    lime_exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for {model_name}", fontsize=14, pad=15)
    plt.xlabel("Probability (Red: Not Stress, Green: Stress)")
    plt.tight_layout()
    plt.show()

#---------------------------------------------Main-----------------------------------------------
# Main Execution
def main():
    print("Starting EEG Stress Analysis Pipeline")
    data_path = r"C:\Users\Raymo\OneDrive\Desktop\Sunway Uni\Classes\Cp2\CodesProject\EEG Stress Project\Data\Data mat\Data\filtered_data"
    train_data, train_labels, test_data, test_labels = load_eeg_data(data_path)
    train_features, train_epoch_labels, feature_names = extract_features(train_data, train_labels, dataset_type="training")
    test_features, test_epoch_labels, _ = extract_features(test_data, test_labels, dataset_type="testing")
    
    # Specify which PCA components to use for visualizations
    pca_x = 5 #PCA_1   
    pca_y = 8 #PCA_2
    
    results = train_and_evaluate(train_features, train_epoch_labels, test_features, test_epoch_labels, 
                                 feature_names, n_components=25, pca_x=pca_x, pca_y=pca_y)
    
    print("\nStep 7: Final Results")
    for model_name, result in results.items():
        print(f"{model_name} Accuracy: {result['accuracy']:.4f}, Best Params: {result['best_params']}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()