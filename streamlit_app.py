import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

st.title("ML Model from CSV")

# File upload widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display dataset shape and first few rows
    st.write("Shape of dataset:", df.shape)
    st.write(df.head())

    # Check if dataset is empty
    if df.shape[0] == 0:
        st.error("⚠️ The file is empty or could not be read properly.")
    else:
        # Input for target column name
        target_column = st.text_input("Enter target column name:")

        if target_column:
            # Check if target column exists in dataset
            if target_column not in df.columns:
                st.error(f"⚠️ The column '{target_column}' does not exist in the dataset.")
            else:
                # Split features (X) and target (y)
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Handle missing values in X
                imputer = SimpleImputer(strategy="mean")  # Replace NaN with column mean
                X = imputer.fit_transform(X)

                # Check if dataset has enough samples
                if len(X) < 2:
                    st.error("⚠️ Not enough samples to train the model.")
                else:
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Initialize and train the Logistic Regression model
                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    # Predict on the test set
                    y_pred = model.predict(X_test)

                    # Calculate accuracy score
                    accuracy = accuracy_score(y_test, y_pred)

                    # Display accuracy
                    st.success(f"✅ Accuracy: {accuracy:.2f}")