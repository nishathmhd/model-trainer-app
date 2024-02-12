import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Function to train and evaluate the model
def train_and_evaluate_model(df, algorithm, hyperparameters):
    X = df.drop(columns=['target'])  # Assuming 'target' is the column to predict
    y = df['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the selected algorithm with hyperparameters
    model = algorithm(**hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return model, accuracy, precision, recall, f1, roc_auc

# Function to perform k-fold cross-validation
def perform_cross_validation(df, algorithm, hyperparameters, cv=5):
    X = df.drop(columns=['target'])
    y = df['target']
    model = algorithm(**hyperparameters)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# Function to display model evaluation metrics
def display_evaluation_metrics(accuracy, precision, recall, f1, roc_auc):
    st.write("### Model Evaluation Metrics")
    st.write(f"- Accuracy: {accuracy:.2f}")
    st.write(f"- Precision: {precision:.2f}")
    st.write(f"- Recall: {recall:.2f}")
    st.write(f"- F1 Score: {f1:.2f}")
    st.write(f"- ROC AUC Score: {roc_auc:.2f}")

# Function to handle errors during model training
def handle_model_training_errors(e):
    st.error(f"An error occurred during model training: {str(e)}")

# Function to display progress feedback during model training
def display_progress_feedback(progress):
    st.write(f"Training progress: {progress}%")
    

# Function to plot data
def plot_data(df):
    st.write("### Data Visualization")

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(numerical_cols) > 0:
        st.write("### Numerical Columns")
        for col in numerical_cols:
            st.write(f"#### {col}")
            fig = px.histogram(df, x=col, title=f'Histogram of {col}')
            st.plotly_chart(fig)

    if len(categorical_cols) > 0:
        st.write("### Categorical Columns")
        for col in categorical_cols:
            st.write(f"#### {col}")
            fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, title=f'Bar Chart of {col}')
            st.plotly_chart(fig)

# Function to create an interactive heatmap
def plot_heatmap(df):
    st.write("### Heatmap")
    corr_matrix = df.corr()
    fig = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.index,
                   y=corr_matrix.columns,
                   colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap', width=800, height=600)
    st.plotly_chart(fig)

# Function to create an interactive scatter plot
def plot_scatter(df):
    st.write("### Scatter Plot")
    x_col = st.selectbox("Select X-axis", options=df.columns)
    y_col = st.selectbox("Select Y-axis", options=df.columns)
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot of {x_col} vs {y_col}')
    st.plotly_chart(fig)

# Function to save the trained model
def download_model(model):
    # Save the model using pickle
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    session_state = st.session_state

    if 'df' not in session_state:
        session_state.df = None

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Model Trainer")
        task = st.radio("Options", ["Upload", "Modelling", "Visualization",])
        

        if os.path.exists("trained_model.pkl"):
            download_button = st.download_button(
                label="Download Trained Model",
                data=open("trained_model.pkl", "rb").read(),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
            
        st.info("This application helps you build and explore your data.")    

    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)     

    if task == "Upload":
        st.title("AutoML")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            st.write("### Dataset")
            df = pd.read_csv(uploaded_file, index_col=None)
            session_state.df = df
            st.dataframe(df)

    if task == "Modelling":
        if session_state.df is None:
            st.error("Please upload a dataset first.")
        else:
            st.write("### Data Wizard")

            # Model Selection
            selected_algorithm = st.selectbox("Select algorithm", ["Random Forest", "Gradient Boosting", "SVM"])
            if selected_algorithm == "Random Forest":
                algorithm = RandomForestClassifier
                default_hyperparameters = {'n_estimators': 100, 'max_depth': 10}
            elif selected_algorithm == "Gradient Boosting":
                algorithm = GradientBoostingClassifier
                default_hyperparameters = {'n_estimators': 100, 'learning_rate': 0.1}
            else:
                algorithm = SVC
                default_hyperparameters = {'kernel': 'rbf', 'C': 1.0}

            # Hyperparameter Tuning
            st.write("#### Hyperparameter Tuning")
            hyperparameters = {}
            for key, value in default_hyperparameters.items():
                hyperparameters[key] = st.slider(f"{key}", min_value=1, max_value=100, value=value)

            # Model Training
            st.write("#### Model Training")
            try:
                model, accuracy, precision, recall, f1, roc_auc = train_and_evaluate_model(session_state.df, algorithm, hyperparameters)
                display_evaluation_metrics(accuracy, precision, recall, f1, roc_auc)
                download_model(model)  # Save the trained model
            except Exception as e:
                handle_model_training_errors(e)

            # Cross-Validation
            st.write("#### Cross-Validation")
            cv_folds = st.slider("Select number of cross-validation folds", min_value=2, max_value=10, value=5)
            cv_score = perform_cross_validation(session_state.df, algorithm, hyperparameters, cv=cv_folds)
            st.write(f"Average cross-validation accuracy: {cv_score:.2f}")

    if task == "Visualization":
        if session_state.df is None:
            st.error("Please upload a dataset first.")
        else:
            plot_data(session_state.df)
            plot_heatmap(session_state.df)
            plot_scatter(session_state.df)

if __name__ == "__main__":
    main()
