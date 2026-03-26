import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def mse_loss(Xb, y, w):
    y_hat = predict(Xb, w)
    return np.mean((y_hat - y) ** 2)


def mse_grad(Xb, y, w):
    n = len(y)

    return (2/n) * (Xb.T @ (Xb @ w - y))




def predict(Xb, w):
    y_hat = Xb @ w
    return y_hat


def add_bias_column(X):
    n = X.shape[0]

    ones = np.ones((n, 1), dtype=X.dtype)

    Xb = np.hstack([ones, X])

    return Xb


def main():
    df = pd.read_csv('Chevy_US_Data.csv')

    SEED = rd.randrange(3320, 4320)

    TARGET_COL = "Total Sales"

    test_size = 0.15
    val_size = 0.15

    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    X_df = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=SEED)

    val_fraction = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_fraction, random_state=SEED)

    from pandas.api.types import is_numeric_dtype

    numeric_features = [c for c in X_df.columns if is_numeric_dtype(X_df[c])]

    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ],
        remainder="drop", )

    full_pipeline = Pipeline(steps=[
        ("preprocessing", pre),
        ("model", LinearRegression())
    ])

    X_train_p = pre.fit_transform(X_train)
    X_val_p = pre.transform(X_val)
    X_test_p = pre.transform(X_test)

    Xb_tr = add_bias_column(X_train_p)
    Xb_va = add_bias_column(X_val_p)

    # Initialize Weights

    w = np.zeros(Xb_tr.shape[1], dtype=np.float64)

    epochs = 1000

    lr = 0.01

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        grad = mse_grad(Xb_tr, y_train, w)

        w = w - (lr * grad)

        train_losses.append(mse_loss(Xb_tr, y_train, w))

        val_losses.append(mse_loss(Xb_va, y_val, w))

        if (epoch + 1) % 100 == 0:
            print(epoch + 1, train_losses[-1], val_losses[-1])

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    plt.close()

    # Now use the full_pipeline for Linear Regression

    full_pipeline.fit(X_train, y_train)
    val_preds = full_pipeline.predict(X_val)

    print("Validation MSE: ", mean_squared_error(y_val, val_preds))

    Xb_test = add_bias_column(X_test_p)
    test_mse = mse_loss(Xb_test, y_test, w)

    print("Test MSE: ", test_mse)


main()
