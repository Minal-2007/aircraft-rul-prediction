# ✈️ Aircraft Engine Remaining Useful Life (RUL) Prediction

A machine learning project that estimates the **Remaining Useful Life (RUL)** of aircraft engines using sensor data. The goal is to support predictive maintenance by identifying potential engine degradation before failure occurs.

## 📌 Project Overview

Aircraft engines operate under demanding conditions, and unexpected failures can lead to costly maintenance and operational disruptions. Traditional maintenance schedules may not fully reflect the actual condition of each engine.

This project explores a data-driven approach to predictive maintenance by analyzing engine sensor measurements and estimating how many operational cycles remain before an engine reaches its end of useful life.

## 🎯 Objectives

* Analyze aircraft engine sensor data to understand degradation patterns.
* Preprocess and prepare time-series data for machine learning.
* Train a model to estimate Remaining Useful Life.
* Present predictions through an accessible dashboard.
* Demonstrate how machine learning can support condition-based maintenance planning.

## ⚙️ Technologies Used

* **Python** — core programming
* **Pandas & NumPy** — data processing and numerical computation
* **Scikit-learn** — machine learning and model evaluation
* **Matplotlib & Seaborn** — data visualization
* **Streamlit** — interactive web dashboard

*The listed libraries should reflect the packages actually used in your implementation.*

## 📊 Dataset

The project is designed around aircraft engine degradation data, such as the **NASA C-MAPSS turbofan engine dataset**, which contains run-to-failure trajectories and sensor measurements collected under different operating conditions.

Typical data fields include:

* Engine unit identifier
* Operational cycle
* Operating settings
* Multiple sensor measurements
* Remaining Useful Life target value

Dataset source: [NASA Prognostics Center of Excellence — Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## 🔄 Project Workflow

1. **Data collection** — load the aircraft engine sensor dataset.
2. **Data preprocessing** — clean the data and organize measurements by engine and operational cycle.
3. **Feature engineering** — prepare relevant sensor and cycle-based features.
4. **Model training** — train a regression model to predict Remaining Useful Life.
5. **Model evaluation** — assess prediction quality using appropriate regression metrics.
6. **Dashboard deployment** — provide an interface for exploring predictions and results.

## 🧠 What Is Remaining Useful Life?

Remaining Useful Life estimates the number of operational cycles an engine can continue operating before reaching a defined end-of-life or failure condition.

For example, an estimated RUL of 25 cycles indicates that the model predicts approximately 25 operating cycles remain under the dataset's assumptions. It is a model estimate, not a guarantee of actual engine life.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🌐 Live Demo

**Streamlit App:** [Add your deployed dashboard URL here]

## 📈 Evaluation

Model performance should be reported using the metrics calculated during testing, such as:

* **MAE (Mean Absolute Error):** average absolute difference between predicted and actual RUL.
* **RMSE (Root Mean Squared Error):** penalizes larger prediction errors more heavily.
* **R² Score:** measures the proportion of target variance explained by the model.

**Actual model and results:** Add your trained model name, test metrics, and any comparison with baseline models here.

## 🔭 Future Improvements

* Compare multiple regression algorithms.
* Improve feature engineering for engine degradation trends.
* Add prediction uncertainty or confidence intervals where appropriate.
* Include sensor trend visualizations and engine-level health summaries.
* Evaluate generalization across different engine operating conditions.
* Explore integration with real-time sensor monitoring systems.

## ⚠️ Limitations

Predictions depend on the dataset, model assumptions, sensor quality, and operating conditions. This project is intended for educational and research purposes and **is not a certified aircraft maintenance or airworthiness system**.

## 👩‍💻 Author

**Minal Gunasekaran**
Mechatronics Engineering | Machine Learning & Predictive Maintenance

* GitHub: [Minal-2007](https://github.com/Minal-2007)

## 📄 License

Add a license file if you intend to distribute this project for reuse. For example, an MIT License may be appropriate if you want to allow broad reuse under its terms.
