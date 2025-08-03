# model_utils.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

def train_model(df, target_column, cat_attribs):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    num_attribs = X.drop(columns=cat_attribs).columns.tolist()
    pipeline = build_pipeline(num_attribs, cat_attribs)
    X_prepared = pipeline.fit_transform(X)
    model = RandomForestRegressor()
    model.fit(X_prepared, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(pipeline, "pipeline.pkl")

    # Save metadata for manual form
    metadata = {
        "features": X.columns.tolist(),
        "cat_attribs": cat_attribs,
        "num_attribs": num_attribs
    }
    joblib.dump(metadata, "metadata.pkl")

    return model, pipeline


def predict_model(df):
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    X_prepared = pipeline.transform(df)
    preds = model.predict(X_prepared)
    df["Predicted_Price"] = preds
    return df
