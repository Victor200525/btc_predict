import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier, plot_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# --- читаем данные ---
df = pd.read_parquet("dataset.parquet")

# --- train / test split по датам ---
train_df = df[df["date"] <= "2021-12-31"]
test_df  = df[df["date"] >= "2022-01-01"]

# --- признаки и target ---
X_train = train_df.drop(columns=["classification_buy", "date", "d", "unixTs"])
y_train = train_df["classification_buy"]

X_test = test_df.drop(columns=["classification_buy", "date", "d", "unixTs"])
y_test = test_df["classification_buy"]

# --- модель ---
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
    objective="binary:logistic",
    eval_metric="logloss"
)

# --- калиброванная модель ---
calibrated_model = CalibratedClassifierCV(
    model,
    method="isotonic",
    cv=5
)

# --- обучение ---
calibrated_model.fit(X_train, y_train)

# --- сохраняем модель ---
joblib.dump(calibrated_model, "xgb_btc_model.pkl")
print("Модель сохранена: xgb_btc_model.pkl")

# --- предсказания ---
pred_class = calibrated_model.predict(X_test)
pred_prob = calibrated_model.predict_proba(X_test)[:,1]

# --- метрики ---
accuracy = accuracy_score(y_test, pred_class)
roc_auc = roc_auc_score(y_test, pred_prob)

print(f"Accuracy: {accuracy:.3f}")
print(f"ROC-AUC : {roc_auc:.3f}")

# --- ROC кривая ---
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--", label="Random guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.show()

# --- важность признаков ---
xgb_model = calibrated_model.calibrated_classifiers_[0].estimator

plot_importance(xgb_model)
plt.title("Feature Importance (XGBoost)")
plt.show()