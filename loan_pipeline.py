import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dagster import job, op, Out, In, repository

# ---------- Ops ----------

@op(out=Out(pd.DataFrame))
def load_data():
    """Load the original dataset"""
    df = pd.read_csv("LoanApprovalPrediction.csv")   # file in same folder
    return df

@op(out=Out(pd.DataFrame))
def load_data_modified():
    """Load the modified dataset"""
    df = pd.read_csv("LoanApprovalPrediction_Modified.csv")   # file in same folder
    return df

@op(ins={"df": In(pd.DataFrame)}, out=Out(pd.DataFrame))
def preprocess(df):
    # Fill missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Encode categorical
    le = LabelEncoder()
    for col in ['Gender','Married','Education','Self_Employed','Property_Area','Dependents','Loan_Status']:
        df[col] = le.fit_transform(df[col].astype(str))

    # Final cleanup: drop any remaining rows with NaNs
    df.dropna(inplace=True)

    return df

@op(ins={"df": In(pd.DataFrame)})
def eda(df):
    sns.countplot(x="Loan_Status", data=df)
    plt.title("Loan Status Distribution")
    plt.show()

    plt.hist(df["ApplicantIncome"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Applicant Income")
    plt.show()

    plt.hist(df["LoanAmount"].dropna(), bins=30, color="salmon", edgecolor="black")
    plt.title("Distribution of Loan Amount")
    plt.show()

@op(ins={"df": In(pd.DataFrame)}, out=Out(dict))
def split_data(df):
    X = df.drop(columns=["Loan_Status","Loan_ID"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

@op(ins={"split": In(dict)}, out=Out(dict))
def train_models(split):
    X_train, X_test, y_train, y_test = (
        split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    )

    results = {}
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    results["Decision Tree"] = (accuracy_score(y_train, dt.predict(X_train)),
                                accuracy_score(y_test, dt.predict(X_test)))

    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7)
    rf.fit(X_train, y_train)
    results["Random Forest"] = (accuracy_score(y_train, rf.predict(X_train)),
                                accuracy_score(y_test, rf.predict(X_test)))

    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = (accuracy_score(y_train, lr.predict(X_train)),
                                      accuracy_score(y_test, lr.predict(X_test)))

    return results

@op(ins={"results": In(dict)})
def plot_results(results):
    models = list(results.keys())
    train_acc = [results[m][0] for m in models]
    test_acc = [results[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10,6))
    bars1 = plt.bar(x - width/2, train_acc, width, label='Train Accuracy', color='purple')
    bars2 = plt.bar(x + width/2, test_acc, width, label='Test Accuracy', color='orange')

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}", ha='center', va='bottom')

    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Jobs ----------

@job
def loan_pipeline_original():
    df = load_data()
    clean = preprocess(df)
    eda(clean)
    split = split_data(clean)
    results = train_models(split)
    plot_results(results)

@job
def loan_pipeline_modified():
    df = load_data_modified()
    clean = preprocess(df)
    eda(clean)
    split = split_data(clean)
    results = train_models(split)
    plot_results(results)

# ---------- Repository ----------
@repository
def loan_repo():
    return [loan_pipeline_original, loan_pipeline_modified]

import time

@job
def loan_pipeline_original():
    start = time.time()
    df = load_data()
    clean = preprocess(df)
    split = split_data(clean)
    results = train_models(split)
    plot_results(results)
    end = time.time()
    print(f"Dagster pipeline (original) execution time: {end - start:.2f} seconds")

@job
def loan_pipeline_modified():
    start = time.time()
    df = load_data_modified()
    clean = preprocess(df)
    split = split_data(clean)
    results = train_models(split)
    plot_results(results)
    end = time.time()
    print(f"Dagster pipeline (modified) execution time: {end - start:.2f} seconds")