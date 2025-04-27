import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)

# Create Stacking Classifier
stacked_model = StackingClassifier(
    estimators=[('rf', rf), ('gbm', gbm)], 
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
)

# Train the stacked model
stacked_model.fit(X_train, y_train)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)
    
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Standardize user input
user_data_scaled = scaler.transform(user_data)

# PREDICTION
user_result = stacked_model.predict(user_data_scaled)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
color = 'red' if user_result[0] == 1 else 'blue'

# Plotting Function
def plot_graph(x, y, user_x, user_y, title, palette, xticks, yticks):
    fig = plt.figure()
    sns.scatterplot(x=X[:, x], y=X[:, y], hue=y, palette=palette)
    plt.scatter(user_x, user_y, color=color, s=150, edgecolors='black')
    plt.xticks(np.arange(*xticks))
    plt.yticks(np.arange(*yticks))
    plt.title(title)
    st.pyplot(fig)

# Graphs
plot_graph(7, 0, user_data['Age'][0], user_data['Pregnancies'][0], 'Pregnancy Count Graph', 'Greens', (10, 100, 5), (0, 20, 2))
plot_graph(7, 1, user_data['Age'][0], user_data['Glucose'][0], 'Glucose Value Graph', 'magma', (10, 100, 5), (0, 220, 10))
plot_graph(7, 2, user_data['Age'][0], user_data['BloodPressure'][0], 'Blood Pressure Value Graph', 'Reds', (10, 100, 5), (0, 130, 10))
plot_graph(7, 3, user_data['Age'][0], user_data['SkinThickness'][0], 'Skin Thickness Value Graph', 'Blues', (10, 100, 5), (0, 110, 10))
plot_graph(7, 4, user_data['Age'][0], user_data['Insulin'][0], 'Insulin Value Graph', 'rocket', (10, 100, 5), (0, 900, 50))
plot_graph(7, 5, user_data['Age'][0], user_data['BMI'][0], 'BMI Value Graph', 'rainbow', (10, 100, 5), (0, 70, 5))
plot_graph(7, 6, user_data['Age'][0], user_data['DiabetesPedigreeFunction'][0], 'DPF Value Graph', 'YlOrBr', (10, 100, 5), (0, 3, 0.2))

# OUTPUT
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 1 else 'You are Diabetic'
st.title(output)

# ACCURACY
y_pred = stacked_model.predict(X_test)
st.subheader('Model Accuracy:')
st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
