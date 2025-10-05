Customer Churn Prediction using Machine Learning

This project predicts whether a customer will churn (leave a service) based on their account and demographic information. The entire pipeline, from data cleaning and model training to deploying an interactive web application, is covered.

 Key Features 

  * Data Preprocessing: Cleaned the Telco Customer Churn dataset, handled missing values, and encoded categorical features.
  * Imbalance Handling: Addressed the imbalanced dataset using the Synthetic Minority Over-sampling Technique (SMOTE) to improve model performance.
  * Model Training: Trained and evaluated several classification models, including `RandomForestClassifier`, `DecisionTreeClassifier`, and `XGBClassifier`.
  * Interactive UI: Built a user-friendly web application with **Streamlit** that allows users to input customer details and receive an instant churn prediction.

 Tech Stack 🛠️

  * **Python**
  * **Data Science Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn
  * **Visualization**: Matplotlib, Seaborn
  * **Web Framework**: Streamlit

 How to Run 🚀

1.  Clone the repository:
    ```bash
    git clone [your-repo-url]
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
