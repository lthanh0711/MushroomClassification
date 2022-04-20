import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# LOAD DATA
@st.cache(persist=True)
def load_data():
    data = pd.read_csv("./mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

# SPLIT TRAIN TEST
@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=["type"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    return x_train, x_test, y_train, y_test

# PLOT METRICS
def plot_metrics(model, metrics_list, x_test, y_test):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Metrix")
        plot_confusion_matrix(model, x_test, y_test)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
# MAIN    
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushroom edible or poisonous ? 🍄")
    placeholder = st.image("./mushroom.jpg")
    st.sidebar.markdown("Are your mushroom edible or poisonous ? 🍄")
    # Load data
    df = load_data()
    # Get train test set
    x_train, x_test, y_train, y_test = split(df)
    # Choose classifier
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    # Hyperparameters
    if classifier == "Support Vector Machine (SVM)":
        # Display in sidebar
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=0.01, step=0.01, key="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
        metrics_list = st.sidebar.multiselect("Metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        
        # Display to main window
        if st.sidebar.button("Classify", key="classify"):
            placeholder.empty()
            st.subheader("Support Vector Machine (SVM)")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(model, metrics_list, x_test, y_test)
            
    if classifier == "Logistic Regression":
        # Display in sidebar
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=0.01, step=0.01, key="C_LR")
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
        metrics_list = st.sidebar.multiselect("Metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        
        # Display to main window
        if st.sidebar.button("Classify", key="classify"):
            placeholder.empty()
            st.subheader("Logistic Regression")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(model, metrics_list, x_test, y_test)
            
    if classifier == "Random Forest":
        # Display in sidebar
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key="n_estimator")
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
        metrics_list = st.sidebar.multiselect("Metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        
        # Display to main window
        if st.sidebar.button("Classify", key="classify"):
            placeholder.empty()
            st.subheader("Random Forest")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred).round(2))
            plot_metrics(model, metrics_list, x_test, y_test)
        
    
    # Checkbox to show raw data
    if (st.sidebar.checkbox("Show raw data", False)):
        st.subheader("Mushroom Dataset for Classification")
        st.write(df)
    

if __name__ == '__main__':
    main()


