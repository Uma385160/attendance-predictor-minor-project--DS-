# attendance-predictor-minor-project--DS-

!pip install gradio plotly scikit-learn pandas matplotlib seaborn --quiet


import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(file):
    df = pd.read_csv(file.name)
    return df.head().to_html()


def train_and_dashboard(file, target_col="Result"):
    
    df = pd.read_csv(file.name)
    
  
    df = df.dropna()
    df = df.copy()
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
  
    X = pd.get_dummies(X, drop_first=True)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
   
    fig1 = px.histogram(df, x=target_col, color=target_col, title="Student Performance Distribution")
    
    
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig2 = px.bar(feature_importances.head(10), x="Importance", y="Feature", orientation="h",
                  title="Top 10 Important Features")
    
 
    pass_rate = (df[target_col].value_counts(normalize=True).get("Pass", 0)) * 100
    
    summary = f" Model Accuracy: {acc*100:.2f}%\n Pass Percentage: {pass_rate:.2f}%"
    
    return summary, fig1, fig2


with gr.Blocks() as demo:
    gr.Markdown("## Student Academic Performance Predictor Dashboard")
    
    with gr.Row():
        file_input = gr.File(label="Upload CSV File")
        output_html = gr.HTML(label="Preview Data")
    
    file_input.change(load_dataset, inputs=file_input, outputs=output_html)
    
    with gr.Row():
        target_col = gr.Textbox(value="Result", label="Target Column (Default: Result)")
        btn = gr.Button("Train Model & Generate Dashboard")
    
    summary = gr.Textbox(label="Model Summary")
    plot1 = gr.Plot(label="Performance Distribution")
    plot2 = gr.Plot(label="Feature Importance")
    
    btn.click(train_and_dashboard, inputs=[file_input, target_col], outputs=[summary, plot1, plot2])


demo.launch(share=True)
  
