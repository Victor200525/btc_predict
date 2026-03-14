import requests
import pandas as pd
import joblib

# --- загрузка модели ---
model = joblib.load("xgb_btc_model.pkl")

# --- API HODL waves ---
url = "https://bitcoin-data.com/api/v1/realized-cap-hodl-waves"
data = requests.get(url).json()

df = pd.DataFrame(data)

# --- признаки ---
features = [
    "age_0d_1d",
    "age_1d_1w",
    "age_1w_1m",
    "age_1m_3m",
    "age_3m_6m",
    "age_6m_1y",
    "age_1y_2y",
    "age_2y_3y",
    "age_3y_4y",
    "age_4y_5y",
    "age_5y_7y",
    "age_7y_10y",
    "age_10y"
]

# --- убираем строки без данных ---
df = df.dropna()

# --- выбираем нужные даты ---
dates_to_check = ["2025-04-25", "2023-10-23"]
df_selected = df[df["d"].isin(dates_to_check)]

# --- прогноз ---
X = df_selected[features].astype(float)
prob = model.predict_proba(X)[:,1]

# --- вывод ---
for date, p in zip(df_selected["d"], prob):
    print(f"Дата: {date} → Вероятность BUY: {p:.2f}")
