
# Loan Approval Prediction Analysis

## Overview
This project involves building and evaluating machine learning models to predict loan approval decisions. The analysis includes various preprocessing techniques such as missing value handling, outlier detection, and multicollinearity removal to ensure high-quality data for training the models. The project implements five machine learning models:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting

## Objective
The goal is to determine the best-performing model for predicting whether a loan should be approved or rejected based on applicant data, using performance metrics like Precision, Recall, F1-Score, and Processing Time.

## Dataset
The dataset contains information about applicants, such as age, income, family size, credit card spending, education level, and more. The target variable is whether the applicant's loan was approved or not.

## Preprocessing Steps
1. **Missing Value Handling**: The dataset was checked for missing values. Any missing values were handled using appropriate imputation techniques.
2. **Outlier Detection**: Log transformation was applied to variables like 'Income' and 'CCAvg' (credit card average spending) to mitigate the effects of outliers.
3. **Multicollinearity Resolution**: Using Variance Inflation Factor (VIF), highly correlated variables like 'Age' and 'Experience' were identified. 'Experience' was removed to reduce multicollinearity.

## Models
The following machine learning models were trained and evaluated:
- **Logistic Regression**: A linear model to predict binary outcomes.
- **SVM (Support Vector Machine)**: A classifier that finds the optimal hyperplane separating different classes.
- **Decision Tree**: A model that splits data into branches to predict the outcome.
- **Random Forest**: An ensemble model that builds multiple decision trees to improve accuracy.
- **Gradient Boosting**: A boosting technique that builds models sequentially to reduce errors.

## Evaluation Metrics
The models were evaluated using the following metrics:
- **Precision**: The accuracy of positive predictions.
- **Recall**: The ability to capture all relevant positive cases.
- **F1-Score**: A balance between precision and recall, giving a holistic view of model performance.
- **Processing Time**: The time it took to train and evaluate each model.

## Results
The Random Forest model was determined to be the best-performing model with an F1-score of 0.947. It balances both precision and recall while maintaining a moderate processing time.

## Repository Structure
- `data/`: Contains the dataset used for the analysis.
- `notebooks/`: Jupyter notebooks used for data preprocessing, model training, and evaluation.
- `reports/`: Includes the final APA-style report.
- `src/`: Contains Python scripts for data preprocessing and model training.
- `README.md`: This file, explaining the project overview and structure.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction-analysis.git
   ```
2. Navigate to the repository:
   ```bash
   cd loan-approval-prediction-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/loan_approval_analysis_final.ipynb
   ```

## Conclusion
The project demonstrates the importance of data preprocessing in improving model performance. The Random Forest model was identified as the best model for predicting loan approval based on applicant features.

## Authors
- Pinkesh Tandel
- Dennis Darko

## Acknowledgments
- Dr. Nina Rajabi Nasab, Northeastern University
