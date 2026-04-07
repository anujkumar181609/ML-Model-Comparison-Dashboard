# 🚀 ML Model Comparison Dashboard

A clean and interactive **Machine Learning Dashboard** built using **Streamlit** that allows users to upload datasets, preprocess data automatically, and compare multiple ML models for both **classification** and **regression problems**.

---

## 📌 Features

### 📊 1. Data Preview

* View dataset in full-width table
* Key metrics:

  * Number of rows
  * Number of columns
  * Missing values
* Column listing
* 🎯 Target column selection (only here for better UX)

---

### 🧹 2. Smart Data Preprocessing

* Removes duplicate rows
* Drops ID-like columns (unique values)
* Converts numeric-like strings to numbers
* Handles missing values:

  * Mean (numerical)
  * Mode (categorical)
* Encodes categorical variables

---

### 🤖 3. Model Training & Comparison

Supports both:

#### 🔹 Classification Models

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes
* Decision Tree
* Random Forest

#### 🔹 Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

### 📈 4. Performance Evaluation

* Automatic problem type detection
* Metrics:

  * Classification → Accuracy, F1 Score
  * Regression → MAE, R² Score
* Best model selection 🏆
* Performance comparison bar chart

---

## ⚙️ Tech Stack

* **Frontend/UI**: Streamlit
* **Data Handling**: Pandas
* **Visualization**: Matplotlib
* **Machine Learning**: Scikit-learn

---

## 🏗️ Project Structure

```
ML Model Comparison Dashboard/
│
├── app.py                 # Main Streamlit app
├── src/
│   ├── preprocess.py      # Data preprocessing logic
│   └── model.py           # Model training & evaluation
│
├── screenshots

```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
cd ml-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Upload a CSV file
2. Select the target column
3. Data gets preprocessed automatically
4. Click **Run Models**
5. Compare results and identify best model

---

## ⚠️ Current Limitations

* No cross-validation (uses train-test split)
* Basic hyperparameter tuning (fixed values)
* Encoding may not be optimal for all datasets
* Limited visualizations

---

## 🚀 Future Improvements

* ✅ Cross-validation for reliable results
* ✅ Hyperparameter tuning (GridSearchCV)
* ✅ Feature importance visualization
* ✅ Confusion matrix & residual plots
* ✅ Model export (download trained model)
* ✅ Deployment (Streamlit Cloud)

---

## 🎯 Learning Outcomes

This project demonstrates:

* End-to-end ML workflow
* Data preprocessing techniques
* Model comparison strategy
* Building interactive ML apps

---

## 🤝 Contributing

Feel free to fork the repo and improve:

* UI/UX
* Model performance
* New features

---

## 📬 Contact

If you found this useful or have suggestions, feel free to connect!

---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub and share it!

---
