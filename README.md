# Credit Card Fraud Detection with Imbalanced Data

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques while addressing the challenges of highly imbalanced data. The dataset used contains 492 fraud cases and 284,315 non-fraud cases, creating a significant imbalance. Two primary machine learning models, Isolation Forest and Random Forest, are employed to detect fraud before and after applying data upsampling techniques.

## Dataset
- **Source**: The dataset used for this project is available on Kaggle and consists of anonymized credit card transactions labeled as fraud or non-fraud.
- **Features**: The dataset consists of 30 columns, including the Class column which labels transactions as 0 (non-fraud) or 1 (fraud).

## Objectives
1. **Analyze and Visualize the Data**: Explore the imbalanced nature of the dataset and visualize class distributions.
2. **Baseline Model (Isolation Forest)**: Implement and evaluate the Isolation Forest algorithm to detect fraudulent transactions in an imbalanced dataset.
3. **Random Forest with SMOTE**: Apply SMOTE (Synthetic Minority Over-sampling Technique) to upsample the minority class and evaluate the performance of the Random Forest model on the balanced data.
4. **Custom Upsampling Method**: Implement a custom upsampling technique based on generating synthetic data using the mean and variance of the minority class and compare its performance with SMOTE.
5. **Cross-Validation**: Ensure the reliability of model performance using cross-validation.

## Methods

### Data Preprocessing
- Handle missing values (if any).
- Split the dataset into training and testing sets.
- Visualize the class imbalance using bar plots.

### Modeling
- **Isolation Forest**: Evaluate the baseline performance of an Isolation Forest model on the imbalanced dataset.
- **Random Forest**: Apply Random Forest to the original dataset, then compare performance after using SMOTE and a custom upsampling technique.

### Upsampling Techniques
- **SMOTE**: Generate synthetic minority class samples by interpolating between existing minority samples.
- **Custom Upsampling**: Create synthetic samples by generating data points based on the mean and variance of each feature in the minority class.

### Evaluation Metrics
- **Accuracy**: The proportion of correct predictions.
- **Confusion Matrix**: Visualize the distribution of true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Measure precision, recall, F1-score, and support for both fraud and non-fraud classes.

## Results

### Baseline Model (Isolation Forest)
The model struggled with identifying fraudulent transactions due to the highly imbalanced data, achieving a high accuracy for non-fraud but poor performance for fraud detection.

### Random Forest + SMOTE
After applying SMOTE, the model showed an improvement in detecting fraud cases, although some synthetic data points might have added noise.

### Custom Upsampling Method
The custom method of generating synthetic data based on feature distribution yielded better results for detecting fraud. The Random Forest model performed significantly better with this approach compared to SMOTE.

### Cross-Validation
Cross-validation was applied to ensure the models' consistency and stability. The Random Forest model using custom upsampling achieved an average accuracy of 99.97% across 5-fold cross-validation, indicating strong model performance.

## Key Insights
- **Customized Upsampling**: Generating synthetic data using the feature distribution (mean and variance) might help maintain important feature relationships, leading to better performance for the Random Forest model compared to SMOTE.
- **SMOTE and Noise**: While SMOTE helps balance the dataset, it might introduce synthetic points that distort feature relationships, especially in complex datasets like credit card fraud detection.

## Future Work
- **Feature Engineering**: Further feature engineering could improve model performance, especially in terms of differentiating fraud from non-fraud transactions.
- **Advanced Models**: Exploring deep learning models such as Variational Autoencoders (VAEs) or GANs to generate more realistic synthetic data for fraud detection.

## Getting Started
To replicate the project, download the dataset and run the `CreditCardFraudDetection_ImbalancedData.ipynb` notebook on Google Colab or a local Jupyter environment. Follow the steps provided for data loading, preprocessing, and model evaluation.

### Requirements
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`

## License
This project is licensed under the **CC0: Public Domain license**.

## Acknowledgements
Special thanks to Sanjai Kumaran for inspiration.
