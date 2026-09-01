# Literature Notes — Week 1

---

## 1. Financial Management System for SMEs: Real-World Deployment of AR & Cash Flow Prediction (arXiv 2511.03631)
**Method:**
An integrated system with two modules: an SVM-based binary classifier for predicting whether an invoice will be significantly delayed, and a modular cash-flow forecasting approach with four specialized sub-models for different income/expense categories. The models use engineered historical payment features such as late-payment ratios, average delays, outstanding invoices, and moving averages.

**Dataset:**
Three real invoice datasets were used: 297 invoices from Cluee/60 customers, 2,466 invoices from the IBM Late Payment Histories dataset, and 12,071 invoices from the Payment Date Dataset. For cash-flow forecasting, because real cash-flow data was limited, the authors also generated synthetic data for 1,000 simulated users over one year, containing 422,306 work sessions.

**Headline metric/finding:**
For accounts-receivable prediction, SVM achieved 0.56 balanced accuracy on the difficult Cluee startup dataset and 0.72 on the IBM dataset; moving-average features improved performance by 3–5 percentage points. For cash-flow forecasting, their method achieved 11.85% MAPE when predicting 11 months from only 1 month of history, compared with 159.40% for Prophet and 166.24% for SVR.

**How this differs from our project:**
This is the closest existing system to ours, and importantly, it already tests one sparse-data scenario (1-month history → 11-month forecast). Our contribution needs to be sharper than "we test scarce data" as a result — it's a systematic study across multiple history lengths producing a robustness curve, not one setting. We also differ by unifying forecasting and risk into a single liquidity-stress probability (rather than separate forecast/classification outputs) with SHAP-based explanation attached, and by adding an early-detection-horizon study — how many days in advance a crisis can be flagged — which this paper doesn't attempt at all.


---

## 2. Literature Review on Optimizing Cash Flow Forecasting Using ML in SMEs (IEEE)

**Method:**
A systematic literature review following the PRISMA framework. The authors reviewed existing research on the application of machine learning to cash-flow forecasting in SMEs and examined its relationship with financial decision-making.

**Dataset:**
No original dataset was collected. The study synthesizes findings from previously published research selected using predefined review criteria.

**Headline metric/finding:**
The review concludes that various ML models can improve cash-flow forecasting by capturing patterns in financial data. Improved forecasting can support better financial decision-making and may help reduce SME insolvency risk.

**How this differs from our project:**
This paper is a literature review rather than a new predictive system. It focuses broadly on improving cash-flow forecasting in SMEs rather than developing an early-warning system for impending liquidity stress. It does not, based on the accessible material, explicitly evaluate prediction lead time, sparse historical-data robustness, or SHAP-based explanations. These are therefore areas that our project investigates more directly.

---

## 3. Corporate Financial Distress Prediction with Machine Learning Techniques (Pizzi et al.)

**Method:**
The study compares machine-learning approaches for predicting corporate financial distress in SMEs. The work considers ML classification models using financial indicators, with Random Forests and Neural Networks being the main ML approaches discussed/evaluated, alongside traditional approaches such as Logistic Regression.

**Dataset:**
A restricted dataset of selected Italian SMEs covering 2017–2021, using firm-level financial indicators. The exact number of SMEs is not stated in the accessible abstract/metadata and should be verified from the results section before recording it.

**Headline metric/finding:**
The study investigates whether machine-learning models can effectively predict future corporate financial distress/failure from financial indicators. The exact best-model performance metric from this chapter remains unverified from the accessible results, so the previously mentioned AUC = 0.94 should NOT be attributed to this paper.

**How this differs from our project:**
The paper focuses on corporate financial distress/failure prediction using firm-level financial indicators. Our project focuses specifically on impending liquidity stress, using transaction-level and temporal cash-flow behavior. Rather than only predicting whether a company will eventually be financially distressed, our objective is to identify an upcoming liquidity problem early, measure the available warning lead time, and explain the prediction using SHAP. This makes our problem more operational and short-term than conventional corporate distress classification.

---

## 4. Interpretable Machine Learning for SME Financial Distress Prediction (Medianovskyi et al., 2023)

**Method:**
The study compares Logistic Regression, Random Forest, XGBoost, CatBoost, and Artificial Neural Networks for SME financial-distress prediction. SHAP is used to interpret and validate the model predictions. CatBoost achieves the best predictive performance among the evaluated models.

**Dataset:**
The study uses SME financial/firm-level information, focusing on variables related to financial condition and company characteristics. The accessible abstract does not provide the exact sample size, so the number of SMEs should be verified from the full paper before recording it.

**Headline metric/finding:**
CatBoost is reported as the most accurate model. SHAP analysis identifies company age, historical overdue accounts receivable, and cash ratio as important factors. SMEs younger than 10 years with historically overdue accounts and a cash ratio below 20% are significantly more likely to experience financial distress. No numerical AUC/accuracy value is stated in the accessible abstract, so it should not be added without verifying the full paper.

**How this differs from our project:**
This is the closest paper to our explainability approach because it combines gradient boosting with SHAP for SME financial-risk prediction. However, it predicts broader financial distress using firm-level financial characteristics rather than short-term liquidity stress from transaction-level cash-flow behavior. It also does not, based on the accessible material, evaluate how many days in advance an upcoming liquidity problem can be detected. Our project therefore combines predictive modeling and SHAP with an explicit early-warning/lead-time objective.

---

## 5. SME Default Prediction: A Systematic Methods Evaluation (Cheraghali & Molnár, 2025)

**Method:**
The study systematically compares 10 machine-learning/statistical methods for SME default prediction: Logistic Regression, Neural Network, Linear Discriminant Analysis, Quadratic Discriminant Analysis, K-Nearest Neighbors, SVM, Decision Tree, XGBoost, LightGBM, and Random Forest. The authors combine these models with different feature-selection and class-rebalancing strategies and evaluate them using strictly separated training and hold-out test samples. The entire analysis is repeated 10 times for robustness.

**Dataset:**
The study uses U.S. SME data from the Compustat database covering 1970–2021. After removing variables with substantial missingness, the final dataset contains 50 financial/firm-level variables and 86,073 observations, including 307 default observations and 85,766 non-default observations. Bankruptcy is defined using Chapter 7/11 filings with a one-year prediction horizon.

**Headline metric/finding:**
Across 6,118 models estimated over 10 iterations, LightGBM achieves the best out-of-sample predictive performance, closely followed by XGBoost and SVM. LightGBM and XGBoost perform best using their built-in feature-selection mechanisms and without class rebalancing. Importantly, excessive oversampling can hurt LightGBM through overfitting. The most frequently selected features include capital employed/total liabilities, EBIT/short-term debt, accounts receivable/total assets, and current ratio.

**How this differs from our project:**
This paper provides strong evidence that gradient-boosted tree models, particularly LightGBM and XGBoost, are effective for SME financial-risk classification. However, it predicts conventional corporate default/bankruptcy using firm-level financial ratios and a one-year horizon. Our project focuses on short-term liquidity stress using transaction-level and temporal cash-flow behavior. We are interested not only in classification performance but also in how early the model can detect an upcoming liquidity problem and why a particular prediction was made using SHAP.

---

## 6. Deep Learning Enhancing Banking Services: A Hybrid Transaction Classification and Cash Flow Prediction Approach (Kotios et al., 2022)

**Method:**
The study develops a hybrid financial-management framework for SMEs with two connected components: transaction categorization and cash-flow forecasting. A rule-based system first categorizes transactions where possible, while a CatBoost classifier handles the remaining uncategorized transactions. The categorized transactions are then transformed into time-series data and used by DeepAR, an RNN-based probabilistic forecasting model, to predict future inflows and outflows. LIME and SHAP are used to explain the transaction-classification predictions.

**Dataset:**
The study uses proprietary real-world banking data provided by the Bank of Cyprus. It contains transaction, customer, and account information for more than 1,000 SMEs from 2017–2020, with over 3.5 million transaction records and more than 40 transaction variables. The dataset includes transaction dates, amounts, debit/credit indicators, descriptions, merchant information, account identifiers, and customer characteristics.

**Headline metric/finding:**
The hybrid transaction categorization model achieves 98% accuracy. SHAP identifies the beneficiary's NACE code and Merchant Category Code as the two most important features, followed by the account key. The framework then aggregates transaction amounts into time-series representations and uses DeepAR to probabilistically forecast SME cash inflows and outflows. The authors also use surrogate data to address sparse time-series and cold-start problems.

**How this differs from our project:**
This is the closest paper so far to our transaction-level approach. It works directly with individual SME banking transactions before aggregating them into account-level time series for cash-flow forecasting, whereas our project uses transaction-level temporal features to predict an explicit liquidity-stress event. Their objective is transaction categorization and cash-flow forecasting, while our objective is early detection of impending liquidity stress. They use SHAP/LIME to explain transaction categorization, whereas we plan to use SHAP to explain the actual liquidity-stress prediction. They also do not evaluate how many days in advance liquidity stress can be detected.

---

## 7. Enterprise Financial Distress Prediction Based on Machine Learning and SHAP Interpretability Analysis (Qi, 2025)

**Method:**
The study develops a machine-learning framework for enterprise financial-distress prediction. It performs data preprocessing, correlation analysis, Recursive Feature Elimination (RFE), model training, and SHAP-based interpretability analysis. Eight models are compared: Decision Tree, Random Forest, Gradient Boosting, XGBoost, Logistic Regression, SVM, KNN, and Naive Bayes. The data are split into 70% training and 30% testing sets, and XGBoost hyperparameters are optimized using grid search and cross-validation.

**Dataset:**
The dataset contains 3,672 observations from 422 companies and 83 financial features. The dataset is highly imbalanced, with distressed companies representing only 3.7% of all observations. SMOTE is therefore used to oversample the minority class.

**Headline metric/finding:**
XGBoost is reported as the best-performing model overall, achieving ROC-AUC = 0.910, accuracy = 0.946, precision = 0.615, recall = 0.150, and F1-score = 0.242. However, XGBoost does NOT have the highest F1-score: Decision Tree achieves 0.319 and Gradient Boosting 0.318. The ROC-AUC result is therefore the strongest evidence for XGBoost's overall ranking. SHAP identifies interest coverage ratio (x36), net profit margin (x44), and current ratio (x13) as the most influential financial features.

**How this differs from our project:**
This paper is highly relevant because it combines XGBoost with SHAP for financial-distress prediction. However, it predicts broader corporate financial distress using firm-level financial ratios rather than short-term liquidity stress from transaction-level cash-flow behavior. Its target is also highly imbalanced, with only 3.7% distressed observations. Our project similarly needs to handle rare liquidity-stress events, but focuses on temporal transaction behavior, explicit liquidity-buffer conditions, prediction lead time, and explaining why an SME is expected to enter liquidity stress.

---

## 8. Survey, Classification and Critical Analysis of the Literature on Corporate Bankruptcy and Financial Distress Prediction (Zhao, Ouenniche & De Smedt, 2024)

**Method:**
A comprehensive survey, classification, and critical analysis of the literature on corporate bankruptcy and financial-distress prediction. The authors organize existing research according to definitions of bankruptcy/distress, prediction models and classifiers, predictive drivers, feature-selection methods, performance evaluation methods, data preprocessing, and other methodological issues. The review identifies six major research streams and uses them to derive future research directions.

**Dataset:**
No original dataset is collected. The study reviews and synthesizes previously published research on corporate bankruptcy and financial-distress prediction across different datasets, countries, markets, models, and methodological approaches.

**Headline metric/finding:**
The review finds a strong shift toward advanced machine-learning and ensemble methods, but concludes that there is no single universally superior methodology because model performance depends on the data, implementation choices, and problem setting. It identifies several important future directions, including improving model interpretability, improving data quality and diversity, handling class imbalance and heterogeneous data, studying macroeconomic effects, and addressing ethical issues and bias.

**How this differs from our project:**
This paper is a broad review of corporate bankruptcy and financial-distress prediction rather than a new predictive model. Its future-research recommendations strongly support several aspects of our project: the need for interpretable predictions, better handling of imbalanced and heterogeneous financial data, and models that are robust across different settings. However, the review primarily concerns firm-level bankruptcy/distress prediction. Our project focuses specifically on short-term liquidity stress using transaction-level temporal cash-flow behavior and adds an explicit early-warning objective: measuring how far in advance liquidity stress can be detected. This short-term liquidity-stress and lead-time perspective is more specific than the broader bankruptcy/distress prediction literature reviewed in this paper.

---

## Synthesis 
**What's already been done that's closest to our idea:**
Two papers are closest in spirit. Paper 1 is closest on objective — it forecasts cash flow and classifies invoice delay, and even tests one sparse-data scenario (1 month → 11-month forecast) — but treats forecasting and risk as separate outputs, with no unified liquidity-risk score and no lead-time analysis. Paper 6 is closest on data granularity — it works directly with individual SME transactions, like we plan to — but its objective is transaction categorization and aggregate cash-flow forecasting, not an explicit liquidity-stress classification with explanation. Papers 4, 5, and 7 establish that gradient-boosted models (LightGBM/XGBoost/CatBoost) combined with SHAP are the strongest, most explainable approach for SME financial-risk prediction — but all three operate at firm-level, using annual/quarterly financial ratios rather than day-to-day transaction behavior, and predict distress over a roughly one-year horizon rather than short-term liquidity stress.

**What gap in this literature does our project actually fill:**
No paper in this review combines all four elements of our approach: (1) transaction-level, day-to-day input data, (2) a unified liquidity-stress risk score rather than separate forecast/classification outputs, (3) an explicit study of how many days in advance the risk can be reliably flagged, and (4) a systematic test of how prediction reliability degrades as available history shrinks. Paper 1 has elements of (1) and a single instance of (4); Paper 6 has (1) but not (2)-(4); Papers 4/5/7 have strong versions of gradient-boosting + SHAP but at firm-level granularity, missing (1), (3), and (4) entirely.

**Any dataset mentioned here worth downloading (beyond Kaggle/IBM):**
Genuinely available: the Payment Date Dataset and the Cluee dataset (297 invoices, 60 customers) from Paper 1 — the Cluee dataset in particular is small and real, which could be useful as a natural (not synthetic) test case for the data-scarcity study, rather than just a training source.

Not realistically obtainable: Compustat (Paper 5) is a paid, institutional-subscription database (WRDS), not something to plan around without confirmed IIITD access. The Bank of Cyprus dataset (Paper 6) is a proprietary industry partnership — not public. Paper 7's dataset (3,672 samples) isn't stated as publicly released. Don't budget time chasing any of these three; note them here just so you remember why they were ruled out if it comes up later.