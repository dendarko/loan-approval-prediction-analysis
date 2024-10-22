
# Loan Approval Prediction Project

## Overview

This project focuses on predicting loan approvals using various machine learning models, including Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, and Gradient Boosting. The dataset used in this project contains customer data such as income, family size, credit card usage, and more. By leveraging predictive modeling, the aim is to help a bank make loan approval decisions efficiently.

## Project Structure

The project is structured as follows:

```
├── data/
│   └── Bank_Personal_Loan_Modelling.xlsx   # Dataset used for analysis
├── notebooks/
│   └── loan_approval_analysis_final.ipynb  # Final Jupyter notebook with full analysis
├── reports/
│   └── Loan_Approval_Final_Report.txt      # APA-formatted report
├── README.md                               # Project instructions and details
```

### Files:

1. **loan_approval_analysis_final.ipynb**: The final Jupyter notebook that contains all steps of the analysis, including data preprocessing, model training, evaluation, and feature importance.
2. **Bank_Personal_Loan_Modelling.xlsx**: The dataset used for the analysis, containing customer data relevant to the loan approval process.
3. **Loan_Approval_Final_Report.txt**: An APA-formatted report summarizing the entire analysis, including the results and conclusions.

## Requirements

To run the code in the Jupyter notebook, the following Python packages are required:

- `pandas`
- `numpy`
- `statsmodels`
- `scikit-learn`
- `time`

You can install the required packages using the following command:

```bash
pip install pandas numpy statsmodels scikit-learn
```

## Data Preprocessing

The dataset used in this project contains the following key features:
- **Age**: Customer's age in completed years.
- **Experience**: Number of years of professional experience.
- **Income**: Annual income of the customer in $000.
- **Family**: Family size of the customer.
- **CCAvg**: Average spending on credit cards per month in $000.
- **Mortgage**: Value of the customer's house mortgage (if any) in $000.
- **Personal Loan**: Target variable (0 = rejected, 1 = approved).

### Handling Multicollinearity and Outliers

- **Multicollinearity**: Multicollinearity was addressed using Variance Inflation Factor (VIF). Features with high VIF (such as Experience) were removed.
- **Outliers**: Log transformations were applied to highly skewed features (Income and CCAvg) to handle outliers.

## Model Training

The following models were trained on the data:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

The dataset was split into training and testing sets (80% training, 20% testing), and the features were standardized before training.

## Model Evaluation

The models were evaluated based on the following metrics:
- **Precision**
- **Recall**
- **F1-Score**
- **Processing Time**

These metrics provide insights into the ability of each model to correctly predict loan approvals while minimizing false positives and false negatives.

### Results Summary

| Model               | Precision | Recall | F1-Score | Processing Time (s) |
|---------------------|-----------|--------|----------|---------------------|
| Logistic Regression  | 0.8246    | 0.4476 | 0.5802   | 0.0291              |
| SVM                 | 0.9630    | 0.4952 | 0.6541   | 0.2043              |
| Decision Tree       | 0.8222    | 0.7048 | 0.7590   | 0.0131              |
| Random Forest       | 0.9412    | 0.6095 | 0.7399   | 0.3765              |
| Gradient Boosting   | 0.9077    | 0.5619 | 0.6941   | 0.4140              |

The **Gradient Boosting** model performed the best, achieving the highest **F1-Score**.

## Feature Importance

The most significant variables in the Gradient Boosting model were:
1. **Income_log**: Highest contributor to loan approval.
2. **Family**: Family size significantly influenced loan approval.
3. **CCAvg_log**: Average credit card spending also played a role in loan approval decisions.

## How to Run the Notebook

To run the analysis:

1. Download the project files.
2. Open the **loan_approval_analysis_final.ipynb** file in Jupyter Notebook or Google Colab.
3. Execute the cells in order to preprocess the data, train the models, and evaluate the results.
4. Review the feature importance and final conclusions at the end of the notebook.

## Conclusion

The **Gradient Boosting** model is recommended for loan approval predictions, offering the best balance between accuracy and processing time. Future improvements can include advanced feature engineering and parameter tuning to further optimize the model's performance.

