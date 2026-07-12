# Ames Housing Price Prediction (DATA6100 Project)

This project predicts house prices in Ames, Iowa using a simple but well-engineered Linear Regression model.  
It was completed as part of my Master’s in Data Science (DATA6100 - Predictive Modelling) at the University of Guelph.

## 📊 Overview
- Cleaned and preprocessed raw housing data (handled missing values and outliers)
- Engineered features such as `TotalSF`, `HouseAge`, and `YearsSinceRemod`
- Encoded categorical and ordinal variables properly
- Used Forward Feature Selection with 5-Fold Cross Validation
- Visualized training vs validation RMSE to detect overfitting

## 🧮 Results
- **Training RMSE:** ~23,800  
- **Test RMSE:** ~24,100  
- **Best model:** 49 selected features  

## 📁 Files
- `DATA_6100_Project1_Final_Submission.ipynb` — main notebook  
- `Project_Summary.pdf` — 3-page report summary  
- `data_description.txt` — description of dataset
- `Housing_Data_Train.csv` — Training Dataset
- `Housing_Data_Test.csv` — Test Dataset
  
## 🧠 Key Learning
A carefully engineered Linear Regression model can achieve strong accuracy through good preprocessing and feature selection — no deep learning required!

---
