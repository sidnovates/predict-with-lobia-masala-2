# Multinomial Classification

# Preprocessing

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

# Models

# 1. Decision Trees

### 1.1 CatBoost

**Method Description:**
CatBoost is a gradient boosting algorithm that handles categorical features automatically.

- **Skewed:** Trained on the natural distribution of the dataset.
- **Non-Skewed:** Trained on a balanced dataset created by upsampling minority classes.

**Results:**

| Dataset Size | Data Distribution | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8113 | 0.7625 |
| 80% | Non-Skewed | 0.8991 | 0.7458 |
| 20% | Skewed | 0.8477 | 0.7508 |
| 20% | Non-Skewed | 0.9284 | 0.7340 |

**Leaderboard Score**
- **Skewed_80:** 0.679
- **Non-Skewed_80:** 0.700

### 1.2 XGBoost

**Method Description:**
XGBoost is an optimized distributed gradient boosting library.

- **Skewed:** Trained on the natural distribution.
- **Non-Skewed:** Trained on a balanced dataset (upsampled).

**Results:**
| Dataset Size | Data Distribution | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.9206 | 0.7450 |
| 80% | Non-Skewed | 0.9559 | 0.7363 |
| 20% | Skewed | 0.9968 | 0.7365 |
| 20% | Non-Skewed | 0.9978 | 0.7173 |

**Leaderboard Score**
- **Skewed_80:** 0.675
- **Non-Skewed_80:** 0.700
- **Non-Skewed_20:** 0.663


### 1.3 Ensemble (XGBoost 0.4 + CatBoost 0.6)

**Method Description:**
This method combines the predictions of XGBoost and CatBoost using a weighted average.

- **Weights:** 0.4 for XGBoost, 0.6 for CatBoost.

**Results:**
| Dataset Size | Data Distribution | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8598 | 0.7554 |
| 80% | Non-Skewed | 0.9287 | 0.7458 |
| 20% | Skewed | 0.9398 | 0.7432 |
| 20% | Non-Skewed | 0.9820 | 0.7339 |

**Leaderboard Score**
- **Non-Skewed_80:** 0.705

### 1.4 Ensemble (XGBoost + CatBoost)

**Method Description:**
This method uses a different weighting scheme or a voting mechanism between XGBoost and CatBoost.

- **Skewed:** Trained on the natural distribution.
- **Non-Skewed:** Trained on a balanced dataset.

**Results:**
| Dataset Size | Data Distribution | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| 80% | Skewed | 0.8774 | 0.7498 |
| 80% | Non-Skewed | 0.9429 | 0.7482 |
| 20% | Skewed | 0.9737 | 0.7428 |
| 20% | Non-Skewed | 0.9903 | 0.7313 |

**Leaderboard Score**
- **Skewed_80:** 0.683
- **Non-Skewed_80:** 0.703
- **Skewed_20:** 0.646
- **Non-Skewed_20:** 0.672

### 1.5 Random Forest

**Method Description:**
This method uses Random forests

**Results:**
| Dataset Size | Data Distribution | Validation Accuracy |
| :--- | :--- | :--- |
| 80% | Skewed | 0.7514 |
| 80% | Non-Skewed | 0.7355 |
| 20% | Skewed | 0.7402 |
| 20% | Non-Skewed | 0.7279 |

**Leaderboard Score**
- **Skewed_80:** 0.655
- **Non-Skewed_80:** 0.693
- **Skewed_20:** 0.629
- **Non-Skewed_20:** 0.664

# 2. Multiclass Logistic Regression

### 2.1 Bayesian Approach (Best)

**Description:**
This method utilizes Bayesian Optimization (specifically using Gaussian Processes) to tune the hyperparameters of the Logistic Regression model, primarily the inverse regularization strength `C`. This approach is generally more efficient than grid search for finding optimal hyperparameters.

**Results:**
| Dataset Config | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| 80% Training (Skewed) | 0.7642 | 0.7580 |
| 80% Training (Non-Skewed) | 0.7319 | 0.7112 |
| 20% Training (Skewed) | 0.7938 | 0.7331 |
| 20% Training (Non-Skewed) | 0.7882 | 0.6972 |

**Leaderboard Score**
- **Skewed_80:** 0.664
- **Non-Skewed_80:** 0.690
- **Skewed_20:** 0.645
- **Non-Skewed_20:** 0.656

### 2.2 Multiclass LR Outlier3 (Best)

**Description:**
This variation applies specific outlier removal thresholds to the training data before fitting the model. By removing extreme values (e.g., in `num_females`, `num_males`), it aims to improve the model's robustness and generalization.

**Results:**
| Dataset Config | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| 80% Training (Skewed) | 0.7614 | 0.7575 |
| 80% Training (Non-Skewed) | 0.7322 | 0.7131 |
| 20% Training (Skewed) | 0.7869 | 0.7470 |
| 20% Training (Non-Skewed) | 0.7835 | 0.7072 |

**Leaderboard Score**
- **Skewed_80:** 0.665
- **Non-Skewed_80:** 0.690
- **Skewed_20:** 0.643
- **Non-Skewed_20:** 0.665

### 2.3 Multiclass LR With 6 K-Means Clustering (Best)

**Description:**
This method incorporates feature engineering using K-Means clustering. It clusters the data into 6 clusters and adds the cluster labels as a new categorical feature, allowing the linear model to capture some non-linear structures in the data.

**Results:**
| Dataset Config | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| 80% Training (Skewed) | 0.7645 | 0.7525 |
| 80% Training (Non-Skewed) | 0.7332 | 0.7156 |
| 20% Training (Skewed) | 0.7864 | 0.7371 |
| 20% Training (Non-Skewed) | 0.7868 | 0.7072 |

**Leaderboard Score**
- **Non-Skewed_80:** 0.690

### 2.4 Multiclass LR With 6 Clustering (Proper) (K-Means + GMM)

**Description:**
An extension of the clustering approach that utilizes both K-Means and Gaussian Mixture Models (GMM) for feature engineering. This likely provides a richer set of cluster-based features to better represent the underlying data distribution.

**Results:**
| Dataset Config | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| 80% Training (Skewed) | 0.7653 | 0.7510 |
| 80% Training (Non-Skewed) | 0.7306 | 0.7156 |
| 20% Training (Skewed) | 0.7923 | 0.7331 |
| 20% Training (Non-Skewed) | 0.7795 | 0.7072 |

**Leaderboard Score**
- **Skewed_80:** 0.667
- **Non-Skewed_80:** 0.688
- **Skewed_20:** 0.642
- **Non-Skewed_20:** 0.658

### 2.5 Random + Bayesian

**Description:**
This method employs a two-stage hyperparameter tuning strategy. It starts with a Randomized Search to explore a wide range of hyperparameters and identify promising regions, followed by Bayesian Optimization to fine-tune the parameters for optimal performance.

**Results:**
| Dataset Config | Train Accuracy | Validation Accuracy |
| :--- | :--- | :--- |
| 80% Training (Skewed) | 0.7620 | 0.7590 |
| 80% Training (Non-Skewed) | 0.7287 | 0.7236 |
| 20% Training (Skewed) | 0.7644 | 0.7530 |
| 20% Training (Non-Skewed) | 0.7711 | 0.7131 |

**Leaderboard Score**
- **Skewed_80:** 0.664
- **Non-Skewed_80:** 0.689
- **Skewed_20:** 0.603
- **Non-Skewed_20:** 0.659

# 3. Neural Networks (MLP)

### 3.1 MLP Classifier

**Description:**
This approach uses Scikit-Learn's `MLPClassifier` with three different architectures:

- **Small:** (50,) hidden units.
- **Medium:** (100, 50) hidden units.
- **Large:** (150, 100, 50) hidden units.

It incorporates K-Means clustering (k=6) as an additional feature. The models are evaluated on two specific split configurations:

- **Split A:** 80% Train, 10% Val, 10% Test.
- **Split B:** 20% Train, 40% Val, 40% Test.

**Results:**

**MLP Small:**
| Dataset Config | Train Dist. | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| 80% Train | Skewed | 0.6725 | 0.6733 |
| 80% Train | Non-Skewed | 0.6582 | 0.6343 |
| 20% Train | Skewed | 0.6649 | 0.6763 |
| 20% Train | Non-Skewed | 0.6608 | 0.6661 |

**Leaderboard Score**
- **Skewed_80:** 0.618
- **Non-Skewed_80:** 0.608
- **Skewed_20:** 0.618
- **Non-Skewed_20:** 0.612

**MLP Medium:**
| Dataset Config | Train Dist. | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| 80% Train | Skewed | 0.6606 | 0.6900 |
| 80% Train | Non-Skewed | 0.6757 | 0.6685 |
| 20% Train | Skewed | 0.6829 | 0.6797 |
| 20% Train | Non-Skewed | 0.6681 | 0.6779 |

**Leaderboard Score**
- **Skewed_80:** 0.607
- **Non-Skewed_80:** 0.627
- **Skewed_20:** 0.606
- **Non-Skewed_20:** 0.621

**MLP Large:**
| Dataset Config | Train Dist. | Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| 80% Train | Skewed | 0.6622 | 0.6582 |
| 80% Train | Non-Skewed | 0.6813 | 0.6749 |
| 20% Train | Skewed | 0.6853 | 0.6873 |
| 20% Train | Non-Skewed | 0.6673 | 0.6747 |

**Leaderboard Score**
- **Skewed_80:** 0.614
- **Non-Skewed_80:** 0.619
- **Skewed_20:** 0.643
- **Non-Skewed_20:** 0.614

### 3.2 MLP with SMOTE

**Description:**
This method uses Keras `Sequential` models with similar architectures (Small, Medium, Large) but applies **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data to handle class imbalance.

- **Small:** Dense(64) -> Dropout.
- **Medium:** Dense(128) -> BN -> Dropout -> Dense(64) -> Dropout.
- **Large:** Dense(256) -> BN -> Dropout -> Dense(128) -> Dropout -> Dense(64) -> Dropout.

**Results:**

| Model Architecture | Dataset Config | Validation Accuracy | Test Accuracy |
| :----------------- | :------------- | :------------------ | :------------ |
| **MLP Small**      | 80% Train      | 0.7267              | 0.7482        |
| **MLP Small**      | 20% Train      | 0.7070              | 0.7137        |
| **MLP Medium**     | 80% Train      | 0.7100              | 0.7315        |
| **MLP Medium**     | 20% Train      | 0.7110              | 0.7137        |
| **MLP Large**      | 80% Train      | 0.7139              | 0.7219        |
| **MLP Large**      | 20% Train      | 0.7072              | 0.7066        |

**Leaderboard Score**
- **Small_80:** 0.704
- **Small_20:** 0.677
- **Medium_80:** 0.687
- **Medium_20:** 0.668
- **Large_80:** 0.678
- **Large_20:** 0.664

### 3.3 MLP with SMOTE+Bayesian

**Description:**
This method advances the Neural Network approach by combining SMOTE for data balancing with Bayesian Optimization (via Optuna). 

Did mlp + smote + bayesian only on small nn as it was only  giving promoising results and others did not give much significant result and training time was too long.

**Results:**

| Validation Accuracy | Test Accuracy |
| :------------------ | :------------------ |
| 0.7506 | 0.7777 |

**Leaderboard Score**
- **Smote+Bayesian:** 0.661

# 4. Support Vector Machines (SVM)

### 4.1 SVM (No Tuning)

**Description:**
This approach evaluates standard SVM kernels (Linear, RBF, Poly, Sigmoid) without hyperparameter tuning. It uses the same preprocessing pipeline (including K-Means clustering with k=6) and split configurations as the other methods.

**Results:**

| Kernel      | Dataset Config | Train Dist. | Validation Accuracy |
| :---------- | :------------- | :---------- | :------------------ |
| **RBF**     | 80% Train      | Skewed      | 0.7418              |
| **RBF**     | 80% Train      | Non-Skewed  | 0.7155              |
| **RBF**     | 20% Train      | Skewed      | 0.7335              |
| **RBF**     | 20% Train      | Non-Skewed  | 0.7048              |

| Kernel      | Dataset Config | Train Dist. | Validation Accuracy |
| :---------- | :------------- | :---------- | :------------------ |
| **Linear**  | 80% Train      | Skewed      | 0.7371              |
| **Linear**  | 80% Train      | Non-Skewed  | 0.7020              |
| **Linear**  | 20% Train      | Skewed      | 0.7281              |
| **Linear**  | 20% Train      | Non-Skewed  | 0.6986              |

| Kernel      | Dataset Config | Train Dist. | Validation Accuracy |
| :---------- | :------------- | :---------- | :------------------ |
| **Sigmoid** | 80% Train      | Skewed      | 0.6566              |
| **Sigmoid** | 80% Train      | Non-Skewed  | 0.5992              |
| **Sigmoid** | 20% Train      | Skewed      | 0.6823              |
| **Sigmoid** | 20% Train      | Non-Skewed  | 0.6299              |

| Kernel      | Dataset Config | Train Dist. | Validation Accuracy |
| :---------- | :------------- | :---------- | :------------------ |
| **Poly**    | 80% Train      | Skewed      | 0.7482              |
| **Poly**    | 80% Train      | Non-Skewed  | 0.7108              |
| **Poly**    | 20% Train      | Skewed      | 0.7307              |
| **Poly**    | 20% Train      | Non-Skewed  | 0.6918              |

**Leaderboard Score**
- **Skewed_80_RBF:** 0.571
- **Non-Skewed_80_RBF:** 0.571
- **Skewed_20_RBF:** 0.554
- **Non-Skewed_20_RBF:** 0.597

- **Skewed_80_Linear:** 0.614
- **Non-Skewed_80_Linear:** 0.690
- **Skewed_20_Linear:** 0.598
- **Non-Skewed_20_Linear:** 0.665

- **Skewed_80_Sigmoid:** 0.614
- **Non-Skewed_80_Sigmoid:** 0.682
- **Skewed_20_Sigmoid:** 0.553
- **Non-Skewed_20_Sigmoid:** 0.655

- **Skewed_80_Poly:** 0.616
- **Non-Skewed_80_Poly:** 0.667
- **Skewed_20_Poly:** 0.621
- **Non-Skewed_20_Poly:** 0.663

### 4.2 SVM (Tuned)

**Description:**
This approach uses Bayesian Optimization (via Optuna) to tune hyperparameters for Linear and RBF kernels.

- **Linear:** Tunes `C`.
- **RBF:** Tunes `C` and `gamma`.

**Results:**

| Kernel     | Dataset Config | Train Dist. | Best Accuracy |
| :--------- | :------------- | :---------- | :------------ |
| **Linear** | 80% Train      | Skewed      | 0.7394        |
| **RBF**    | 80% Train      | Skewed      | 0.7514        |
| **Linear** | 80% Train      | Non-Skewed  | 0.7203        |
| **RBF**    | 80% Train      | Non-Skewed  | 0.7171        |
| **Linear** | 20% Train      | Skewed      | 0.7315        |
| **RBF**    | 20% Train      | Skewed      | 0.7382        |
| **Linear** | 20% Train      | Non-Skewed  | 0.7034        |
| **RBF**    | 20% Train      | Non-Skewed  | 0.7175        |

**Leaderboard Score**
- **Skewed_80_RBF:** 0.634
- **Non-Skewed_80_RBF:** 0.686
- **Skewed_20_RBF:** 0.621
- **Non-Skewed_20_RBF:** 0.656

- **Skewed_80_Linear:** 0.616
- **Non-Skewed_80_Linear:** 0.680
- **Skewed_20_Linear:** 0.598
- **Non-Skewed_20_Linear:** 0.665

---

# 🧠 Conclusion

## 🏆 Best Model Performance
- Tree-based **Gradient Boosting models** demonstrated the strongest results on the Kaggle leaderboard.
  - **CatBoost (Skewed 80% Train)** → 0.679
  - **XGBoost (Skewed 80% Train)** → 0.675
- The **top-ranked leaderboard submission** was achieved using an **ensemble approach**:
  - **XGBoost 0.4 + CatBoost 0.6 (Balanced 80% Train)** → **0.705** (Best Overall)
- **Neural Networks (MLP)** were competitive only at small scale:
  - **MLP Small + SMOTE (80% Train)** → 0.704 leaderboard score
  - Further tuning using SMOTE + Bayesian increased test accuracy (0.7777) but resulted in a lower leaderboard gain (0.661), showing that boosting generalizes better for leaderboard evaluation.
- **SVM models**, even when tuned, achieved moderate scores (best 0.634) and were more sensitive to data imbalance and training size.
- **Random Forest** lagged behind boosting and small MLP models, especially on reduced or balanced splits.

**Overall Ranking Trend:**
Boosting Ensemble > CatBoost > XGBoost > MLP Small (Balanced 80%) > SVM (Tuned) > Random Forest > Larger MLPs


---

## ⚖️ Impact of Skewed vs. Balanced Training Data
- **Balancing the training data improved generalization on Kaggle**, especially for:
  - SVM (Linear kernel improved from 0.614 → 0.690 when balanced)
  - Boosting Ensembles (best performer at 0.705)
- **Validation accuracy slightly dropped for boosting when balanced**, indicating synthetic noise from upsampling, but leaderboard performance still improved.
- **Medium & Large MLP networks did not gain significant leaderboard benefits from balancing** and incurred much longer training time.
- **SMOTE was more effective than simple upsampling**, especially for small neural networks, but still could not outperform boosted tree generalization at larger scales.

**Key Observation:**  
✔ Balancing helps fairness and leaderboard performance  
⚠ But may slightly reduce raw validation accuracy due to synthetic noise

---

## 📏 Impact of Dataset Size (80% vs. 20% Train)
- **80% training allowed better learning for most models**, especially boosting and small MLPs.
- **20% training significantly reduced performance**, but:
  - **XGBoost, CatBoost, and Ensembles remained resilient**, still scoring within ~0.66–0.68 range on Kaggle.
  - **Small MLP showed stability**, maintaining nearly the same Kaggle score for 80% and 20% training (~0.618 both skewed, ~0.612 balanced).
  - **SVM performance dropped the most** under limited training data.

**Key Observation:**  
🌳 Boosted trees degrade gracefully when data is limited  
📉 SVM and larger MLPs struggle when train size is extremely small

---

## 🎯 Final Takeaways
| Model Category | Leaderboard Strength | Data Balancing Benefit | Low-Resource (20% Train) Robustness |
|---|---|---|---|
| Gradient Boosting (Tree models) | **Highest** | **High** but slight val drop | **Most Robust** |
| Ensemble (XGBoost + CatBoost) | **Best Overall (0.705)** | **Maximum Benefit** | **Highly Resilient** |
| MLP Neural Network | **Good only at Small scale** | **Helped small NN most** | **Stable only for small NN** |
| SVM | **Moderate even when tuned** | **Largest Benefit from balancing** | **Least Robust** |
| Random Forest | **Lower compared to Boosting & small NN** | **Minor/Negative** | **Low** |

---

# ✅ Final Conclusion
> **The ensemble of XGBoost and CatBoost trained on 80% balanced data generalized best on Kaggle’s leaderboard, while individual gradient-boosted decision trees remained the most robust to both class skew and dataset size reduction. Data balancing improved generalization and fairness, particularly for SVM and ensemble models, but introduced slightly lower validation accuracy in some cases. Training on only 20% of data caused performance degradation for most algorithms, except boosted trees and small MLP networks, which showed relative stability. Overall, gradient-boosting ensembles proved to be the best choice for leaderboard success and real-world multinomial classification generalization.**
