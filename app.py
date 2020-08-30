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

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(r"C:\Users\suraj\Documents\ineuron_DL\coursera\guided_project\mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])

        return data

    

    @st.cache(persist=True)
    def split(df):
        X = df.drop('class',axis=1)
        y = df['class']
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)
        return  X_train,X_test,y_train,y_test

    # @st.cache(persist=True)
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model,X_test,y_test,display_labels=class_names)
            st.pyplot() 

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model,X_test,y_test)
            st.pyplot() 
    
        if 'Precision Recall Curve' in metrics_list:
            st.subheader('PR Curve')
            plot_precision_recall_curve(model,X_test,y_test)
            st.pyplot() 
    
    class_names = ['edible','poisonous']
    df = load_data()
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset classification")
        st.write(df)
    X_train,X_test,y_train,y_test = split(df)

    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier',('SVM','Logistic Regression','Random Forest'))

    if classifier == 'SVM':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C(Regularization Parameter)', 0.01,10.0,step = 0.01, key = 'C')
        kernel = st.sidebar.radio('Kernel',('rbf','linear'))
        gamma = st.sidebar.radio('Gamma',('scale','auto'))

        metrics = st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button('Classify',key='classify'):
            st.subheader('SVM Results')
            model = SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy: ', accuracy.round(2))
            st.write('Recall: ', recall_score(y_test,y_pred,labels=class_names).round(2))
            st.write('Precision: ', precision_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

if __name__ == '__main__':
    main()


