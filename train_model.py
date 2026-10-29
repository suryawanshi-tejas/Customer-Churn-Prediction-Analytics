import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE=Path(__file__).resolve().parent
df=pd.read_csv(BASE/"data"/"customers.csv")
X=df.drop(columns=["churn"])
y=df["churn"]

numeric=["senior_citizen","tenure_months","monthly_charges","total_charges"]
categorical=[c for c in X.columns if c not in numeric]

preprocessor=ColumnTransformer([
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                      ("scaler",StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                      ("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

model=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"))
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,pred))

(BASE/"models").mkdir(exist_ok=True)
joblib.dump(model,BASE/"models"/"churn_model.pkl")
print("Saved:",BASE/"models"/"churn_model.pkl")
