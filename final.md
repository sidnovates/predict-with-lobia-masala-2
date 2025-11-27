# Binomial Classification

## Preprocessing and Exploratory Data Analysis

We analyzed the dataset using scatter plots and box plots to identify potential outliers. Upon visual inspection, no significant outliers were detected in the data. Additionally, we examined the covariance between features and found it to be very low. Consequently, no features were removed, and no specific preprocessing steps such as outlier rejection or feature selection were performed.

### Feature Preprocessing Pipeline
To prepare the data for modeling, we implemented a `ColumnTransformer` to apply specific transformations to numeric and categorical features:

-   **Target Variable**: `RiskFlag`
-   **Numeric Features**: The following 9 features were scaled using **StandardScaler** to ensure zero mean and unit variance:
    -   `ApplicantYears`, `AnnualEarnings`, `RequestedSum`, `TrustMetric`, `WorkDuration`, `ActiveAccounts`, `OfferRate`, `RepayPeriod`, `DebtFactor`
-   **Categorical Features**: The following 7 features were encoded using **OneHotEncoder** (with `handle_unknown="ignore"` and `sparse_output=False`):
    -   `QualificationLevel`, `WorkCategory`, `RelationshipStatus`, `FamilyObligation`, `OwnsProperty`, `FundUseCase`, `JointApplicant`

# Model Training Methodology

## Data Splitting Strategy
To rigorously evaluate model performance, we employed two distinct data splitting strategies:

1.  **Split A (80/10/10)**:
    -   **Training**: 80% of the dataset.
    -   **Validation**: 10% of the dataset.
    -   **Testing**: 10% of the dataset.
    -   *Purpose*: To assess performance when ample training data is available.

2.  **Split B (20/40/40)**:
    -   **Training**: 20% of the dataset.
    -   **Validation**: 40% of the dataset.
    -   **Testing**: 40% of the dataset.
    -   *Purpose*: To simulate low-resource scenarios and test model generalization with limited training data.

## Data Distribution (Skewed vs. Non-Skewed)
We also experimented with the class distribution of the training data:

-   **Skewed (Natural Distribution)**: The dataset is used as-is, preserving the original class imbalance. This reflects the real-world distribution of the data.
-   **Non-Skewed (Balanced)**: We applied upsampling to the minority classes to match the size of the majority class. This technique aims to prevent the model from becoming biased toward the majority class.

## Experimental Configurations
For every model architecture, we trained and evaluated four distinct configurations:
1.  **80% Train - Skewed**
2.  **80% Train - Non-Skewed**
3.  **20% Train - Skewed**
4.  **20% Train - Non-Skewed**

This allows us to analyze the impact of both dataset size and class balance on model performance.


# MODELS

## 1. Standard Binomial Logistic Regression

**Method Description:**
A baseline Logistic Regression model is trained using the `lbfgs` solver with a maximum of 500 iterations. It models the probability of the binary outcome using the logistic function. 

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8854 | 0.8855 |0.885
| 80% | Non-Skewed | 0.6746 | 0.6760 |0.677
| 20% | Skewed | 0.8848 | 0.8855 |0.885
| 20% | Non-Skewed | 0.6788 | 0.6714 |0.676

## 2. Binomial Logistic Regression with Bayesian Optimization

**Method Description:**
This approach utilizes **Bayesian Optimization**  to exhaustively search for the optimal hyperparameters .  The single set of "best" hyperparameters identified from this process was then applied to train models for all four data scenarios, testing the generalizability of the tuned configuration.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy | Leaderboard
| :--- | :--- | :--- | :--- | :--- 
|80% | Skewed | 0.8854 | 0.8855 | 0.885
| 80% | Non-Skewed | 0.6745 | 0.6761 | 0.677
| 20% | Skewed | 0.8852 | 0.8856 | 0.885
| 20% | Non-Skewed | 0.6775 | 0.6712 | 0.676

## 3. Tuned Binomial Logistic Regression (Bayesian Approach + Random Search)

**Method Description:**
This method employs a robust two-stage hyperparameter tuning process to optimize the regularization strength (`C`). First, a **Random Search** explores a wide logarithmic range of values. This is followed by **Bayesian Optimization** (using Gaussian Process via `skopt`), which refines the search around the most promising region found in the first stage. This tuning process is performed **independently** for each of the four data scenarios to find the specific optimal parameters for that distribution.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.6744 | 0.6760 |0.677
| 80% | Non-Skewed | 0.6744 | 0.6760 |0.677
| 20% | Skewed | 0.6786 | 0.6715 |0.676
| 20% | Non-Skewed | 0.6786 | 0.6715 |0.676

_Note: The similar results across skewed and non-skewed versions in this method are because the optimal `class_weight='balanced'` found during the 80% Non-Skewed tuning was enforced on all models._


## 5. Random Forest 

**Method Description:**
A Random Forest Classifier is trained with 400 estimators. The implementation (`RandomForest.ipynb`) uses a standard pipeline with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features. For the non-skewed (balanced) configuration, `class_weight='balanced'` is used.
_Note: This specific run (`RF2`) only contains results for the 80% dataset splits._

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8856 | 0.8854 |0.886
| 80% | Non-Skewed | 0.8847 | 0.8844 |0.885

## 7. XGBoost

**Method Description:**
An XGBoost Classifier is trained with `n_estimators=400`, `learning_rate=0.1`, and `max_depth=6`. The implementation (`XGBoost.ipynb`) handles class imbalance for the non-skewed (balanced) configuration by calculating and setting the `scale_pos_weight` parameter (ratio of negative to positive samples). For skewed data, `scale_pos_weight` is set to 1.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8852 | 0.8859 |0.887
| 80% | Non-Skewed | 0.8852 | 0.8859 |0.887
| 20% | Skewed | 0.8831 | 0.8843 |0.885
| 20% | Non-Skewed | 0.8831 | 0.8843 |0.885

_Note: The XGBoost results are identical for Skewed and Non-Skewed configurations._

## 8. XGBoost (AUC Scoring)

**Method Description:**
This variation of the XGBoost implementation utilizes **AUC (Area Under the Curve)** as the evaluation metric during training, likely for early stopping or model selection, instead of the default logloss or error rate. The underlying model architecture and hyperparameters remain similar to the standard XGBoost method.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8852 | 0.8859 | 0.887
| 80% | Non-Skewed | 0.8852 | 0.8859 | 0.887
| 20% | Skewed | 0.8831 | 0.8843 | 0.885
| 20% | Non-Skewed | 0.8831 | 0.8843 | 0.885

_Note: The results for XGBoost (AUC) are identical to the standard XGBoost method (using F1 Score as evaluation metric), indicating that optimizing for AUC in this context yielded the same final accuracy performance._

## 9. Neural Networks

### 9.1 MLP (Multi-Layer Perceptron)

**Method Description:**
The MLP approach involves training fully connected Feed-Forward Neural Networks. Three distinct architectures were defined to test the impact of model complexity:

- **NN_Small:** A simple network with 1 hidden layer (64 units, ReLU) and Dropout (0.2).
- **NN_Medium:** A moderate network with 2 hidden layers (128, 64 units), Batch Normalization, and Dropout (0.3).
- **NN_Deep:** A complex network with 3 hidden layers (256, 128, 64 units), Batch Normalization, and Dropout (0.4/0.3/0.2).

All models use the Adam optimizer (`lr=1e-3`) and Binary Cross-Entropy loss. 

**Results:**

**NN_Small:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8871 | 0.8877 |0.887
| 80% | Non-Skewed | 0.6997 | 0.6998 |0.699
| 20% | Skewed | 0.8848 | 0.8859 |0.887
| 20% | Non-Skewed | 0.7019 | 0.7048 |0.704

**NN_Medium:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8871 | 0.8869 |0.886
| 80% | Non-Skewed | 0.7054 | 0.7092 |0.706
| 20% | Skewed | 0.8841 | 0.8861 |0.886
| 20% | Non-Skewed | 0.8185 | 0.8184 |0.818

**NN_Deep:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8867 | 0.8871 |0.887
| 80% | Non-Skewed | 0.6852 | 0.6858 |0.685
| 20% | Skewed | 0.8837 | 0.8856 |0.885
| 20% | Non-Skewed | 0.6850 | 0.6936 |0.691

### 9.2 MLP with Bayesian Optimization (Tuned)

**Method Description:**
This method applies Bayesian Optimization (using `keras-tuner`) to find the optimal hyperparameters for the MLP architectures. The tuning search space includes the number of units, dropout rates, learning rate, and batch size.
_Note: Results were only found for the **NN_Small** architecture on Skewed datasets as the **NN_Medium** and **NN_Deep** architectures were taking too long to train using Bayesian Tuning._

**Results:**
| Dataset Size | Data Distribution | Model | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | NN_Small | 0.8868 | 0.8871 |0.887
| 20% | Skewed | NN_Small | 0.8837 | 0.8859 |0.887

### 9.3 SMOTE (Synthetic Minority Over-sampling Technique)

**Method Description:**
This approach addresses class imbalance by using SMOTE to oversample the minority class in the training data _before_ training the neural networks. The same three architectures (Small, Medium, Deep) are used.

**Skewed Dataset:**
The dataset was imbalanced, so SMOTE was directly applied to address the minority class and create a more balanced distribution.
No class weights were used, as the application of SMOTE already balanced the class proportions.

**Results:**

**NN_SMOTE_Small:**
| Dataset Size  | Validation Accuracy | Test Accuracy |Leaderbaord
| :--- | :--- | :--- | :--- |
| 80% |  0.7974 | 0.8031 |0.798
| 20% |  0.7599 | 0.7653 |0.765

**NN_SMOTE_Medium:**
| Dataset Size |  Validation Accuracy | Test Accuracy |Leaderbaord
| :--- | :--- | :--- | :--- |
| 80% |  0.8710 | 0.8725 |0.874
| 20% |  0.8328 | 0.8435 |0.852

**NN_SMOTE_Deep:**
| Dataset Size |  Validation Accuracy | Test Accuracy |Leaderboard
| :--- |  :--- | :--- | :--- |
| 80% | 0.8780 | 0.8787 |0.878
| 20% | 0.8381 | 0.8390 |0.840

## 10. Support Vector Machines (SVM)

### 10.1 Linear SVM

**Method Description:**
The Linear SVM model is implemented using `LinearSVC`. It attempts to separate the classes with a linear hyperplane.

- **Skewed:** Trained with standard parameters (`C=1.0`).
- **Non-Skewed:** Trained with `class_weight='balanced'` to handle class imbalance.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8837 | 0.8837 |0.884
| 80% | Non-Skewed | 0.6725 | 0.6738 |0.675
| 20% | Skewed | 0.8837 | 0.8837 |0.885
| 20% | Non-Skewed | 0.6766 | 0.6698 |0.674

### 10.2 RBF SVM (Radial Basis Function)

**Method Description:**
This method uses the `SVC` class with the RBF kernel (`kernel='rbf'`). RBF is a popular kernel that can handle complex non-linear relationships by measuring the similarity between data points.

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8838 | 0.8837 |0.884
| 80% | Non-Skewed | 0.6997 | 0.7002 |0.698
| 20% | Skewed | 0.8837 | 0.8837 |0.884
| 20% | Non-Skewed | 0.7030 | 0.7125 |0.710

### 10.3 Polynomial SVM

**Method Description:**
This method uses the `SVC` class with a polynomial kernel (`kernel='poly', degree=3`). It maps the input features into a higher-dimensional space to find a non-linear decision boundary.
_Note: Results were only found for the **80% Skewed** configuration in the artifacts as others were taking too long to train._

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8837 | 0.8837 |0.884


### 10.4 Sigmoid SVM

**Method Description:**
This method uses the `SVC` class with the Sigmoid kernel (`kernel='sigmoid'`). The Sigmoid kernel is equivalent to a two-layer perceptron neural network.
_Note: Results were only found for a **20% Training Data** configuration as others were taking too long to train and also was not giving good results._

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy | Test Accuracy |Leaderboard
| :--- | :--- | :--- | :--- | :--- |
| 20% | Skewed | 0.5596 | 0.5598 |0.563

# Multinomial Classification

## 1. Preprocessing

The preprocessing pipeline was designed to handle the complexity of the multinomial classification task, ensuring data quality and creating informative features for the neural network models.

### 1.1 Initial Data Cleaning and Null Handling
- **Target Variable Cleaning**: We identified and removed 34 rows where the target variable `spend_category` was null, as these samples cannot be used for supervised learning.
- **String Sanitization**: All object-type columns were processed to strip leading/trailing whitespace and remove trailing commas, ensuring consistency in categorical values.
- **Binary Feature Standardization**: 
    - Columns containing "Yes"/"No" values (e.g., `is_first_visit`, `food_included`, `intl_transport_included`) were mapped to binary integers (1/0).
    - The `has_special_requirements` column was converted to a binary flag: 0 if "None", "NaN", or empty; 1 otherwise.

### 1.2 Ordinal Encoding of Range Features
To preserve the order inherent in range-based features, we applied specific ordinal mappings instead of one-hot encoding:
- **`days_booked_before_trip`**: Mapped to an ordinal scale of 1-6:
    - "1-7" → 1, "8-14" → 2, "15-30" → 3, "31-60" → 4, "61-90" → 5, "90+" → 6.
- **`total_trip_days`**: Mapped to an ordinal scale of 1-4:
    - "1-6" → 1, "7-14" → 2, "15-30" → 3, "30+" → 4.
- **Imputation Strategy**: Missing values in these ordinal columns (and other categorical features) were imputed using the **mode** (most frequent value) to maintain data integrity without introducing synthetic noise.

### 1.3 Outlier Removal
We performed a rigorous outlier analysis and removal process based on domain knowledge and distribution tails. Approximately 71 rows were removed in total based on the following thresholds:
- **`num_females`**: Removed records with > 10 females (28 rows).
- **`num_males`**: Removed records with > 10 males (8 rows).
- **`mainland_stay_nights`**: Removed trips exceeding 90 nights (27 rows).
- **`island_stay_nights`**: Removed trips exceeding 60 nights (8 rows).

### 1.4 Advanced Feature Engineering: Clustering
To capture complex, non-linear relationships between traveler attributes, we employed unsupervised learning as a feature engineering step:
- **Algorithm**: K-Means Clustering.
- **Input Features**: Standardized numeric features (`num_females`, `num_males`, `mainland_stay_nights`, `island_stay_nights`, `days_booked_before_trip_ord`, `total_trip_days_ord`).
- **Configuration**: `n_clusters=6`, `random_state=42`.
- **Optimal K Selection**: We determined the optimal number of clusters (K=6) using a combination of quantitative and visual methods:
    - **Elbow Method**: Analyzed the Within-Cluster Sum of Squares (Inertia) plot to identify the "elbow" point where variance reduction slows down.
    - **Silhouette Analysis**: Calculated Silhouette Scores for K ranging from 2 to 10. The score peaked at K=6, indicating the best separation and cohesion.
    - **PCA Visualization**: Projected the data into 2D space using Principal Component Analysis (PCA) to visually confirm the distinctness of the 6 clusters.
- **Output**: A new categorical feature `kmeans_cluster` was assigned to each sample, representing the "traveler profile" cluster it belongs to. This feature allows the downstream classifier to learn cluster-specific patterns.

### 1.5 Final Data Transformation Pipeline
The final dataset was processed using a `ColumnTransformer` before feeding into the models:
- **Numeric Features**: Standardized using `StandardScaler` to ensure zero mean and unit variance, crucial for Neural Network convergence.
- **Categorical Features**: Encoded using `OneHotEncoder` (including the generated `kmeans_cluster` feature), with `handle_unknown='ignore'` to robustly handle unseen categories in test data.
- **Class Imbalance Handling**: We applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set. This generated synthetic samples for minority classes in the `spend_category` target, ensuring the model does not become biased toward the majority class.

