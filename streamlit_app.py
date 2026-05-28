from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR


TARGET = "Ortalama ΔθTO"
TARGET_LABEL = "Ortalama ΔθTO"
SHEET_NAME = "Puant Raporu"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ADANA_FILE_NAME = "Adana Sıcaklık Trafolar.xlsx"
MERSIN_FILE_NAME = "Mersin Sıcaklık Trafolar.xlsx"
DEFAULT_ADANA_PATH = r"C:\Users\firat\Desktop\tez\proje\Adana Sıcaklık Trafolar.xlsx"
DEFAULT_MERSIN_PATH = r"C:\Users\firat\Desktop\tez\proje\Mersin Sıcaklık Trafolar.xlsx"
CACHE_DIR = APP_DIR / ".streamlit_cache"
PREPARED_CACHE_PATH = CACHE_DIR / "prepared_data.pkl"
RANDOM_STATE = 42
RF_ESTIMATORS = 80
SVM_C = 100.0
SVM_GAMMA = 0.1
SVM_EPSILON = 0.1
SVM_MAX_ROWS = 4_000
LSTM_EPOCHS = 20
CNN_GRU_EPOCHS = 20
BATCH_SIZE = 32


class NeuralModelUnavailable(RuntimeError):
    pass

MONTH_NUMBER_MAP = {
    "Ocak": 1,
    "Şubat": 2,
    "Mart": 3,
    "Nisan": 4,
    "Mayıs": 5,
    "Haziran": 6,
    "Temmuz": 7,
    "Ağustos": 8,
    "Eylül": 9,
    "Ekim": 10,
    "Kasım": 11,
    "Aralık": 12,
}

INITIAL_DROP_COLUMNS = [
    "Trafo",
    "Sınıflandırılmış Trafo",
    "Sınıflandırılmış Merkez",
    "Merkez Kodu",
    "Trafo ΔθTO>90",
]

MODEL_DROP_COLUMNS = [
    "ΔθHS",
    "ΔθHO",
    "ΔθTA",
    "ΔθTO",
    "n Faktörü",
    "m Faktörü",
    "Anlık Kapasite Tarihi",
    "Maksimum Zamanı_Ptot",
    "Minimum Zamanı_Ptot",
    "Maksimum Zamanı_Stot",
    "Minimum Zamanı_Stot",
    "Ortalama Sıcaklık",
    "K ( Yük Faktörü)",
    "R Faktörü",
    "Kurulu Güç",
    "Ortalama_Ptot",
    "Isı Aktarımı (Q)",
    "Ortalama Aktif Güç (ΔθTO>=90)",
    "F_AA(Yaşlanma Faktörü)",
    "Ömür Kat Sayısı",
    "Durum_❌ Çok tehlikeli! İzolasyon bozulması riski",
    "Durum_🟠 Orta seviye yaşlanma",
    "Durum_🟢 Uzun ömür",
    "Durum_🟢 Yüksek verim, uzun ömür",
    "Ortalama Rüzgar Hızı (m/s)",
    "Hedef Yükseklik Hızı",
    "Dış Konveksiyon Katsayısı (h)",
    "İyileşme Faktörü (h/h0)",
    "Boşta Kayıplar (Po) Watt",
    "Yükte Kayıplar (Pk) Watt",
    "Total Kayıp (Ploss)",
    "Rüzgar ΔT Azalma Yüzdesi (%)",
    "A (Yüzey Alanı)",
]

NUMERIC_INPUTS = [
    "Anlık Kapasite",
    "Maksimum_Ptot",
    "Minimum_Ptot",
    "Ortalama_Stot",
    "Maksimum_Stot",
    "Minimum_Stot",
    "Enlem ",
    "Boylam",
    "Ortalama Rüzgar Hızı (kph)",
]

MODEL_OPTIONS = ["Random Forest", "SVM", "LSTM", "CNN-GRU", "Tüm Modeller"]
MODEL_COLORS = {
    "Random Forest": "#d62728",
    "SVM": "#008080",
    "LSTM": "#228b22",
    "CNN-GRU": "#663399",
}


@dataclass
class ClassicalModel:
    model: Any
    features: list[str]
    mae: float
    r2: float
    scaler_x: Any | None = None
    scaler_y: Any | None = None


@dataclass
class NeuralModel:
    model: Any
    features: list[str]
    mae: float
    r2: float
    scaler_x: MinMaxScaler
    scaler_y: MinMaxScaler


def _safe_drop(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=[col for col in columns if col in df.columns])


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_signature(adana_path: Path, mersin_path: Path) -> dict[str, Any]:
    return {
        "adana": file_signature(adana_path),
        "mersin": file_signature(mersin_path),
    }


def resolve_data_paths() -> tuple[str, str]:
    repo_adana = DATA_DIR / ADANA_FILE_NAME
    repo_mersin = DATA_DIR / MERSIN_FILE_NAME
    if repo_adana.exists() and repo_mersin.exists():
        return str(repo_adana), str(repo_mersin)
    return DEFAULT_ADANA_PATH, DEFAULT_MERSIN_PATH


def parse_turkish_month_series(series: pd.Series) -> pd.Series:
    text_values = series.astype("string").str.strip()
    month_pattern = "|".join(MONTH_NUMBER_MAP)
    extracted = text_values.str.extract(rf"(?P<month>{month_pattern})\s+(?P<year>\d{{4}})")
    month_numbers = extracted["month"].map(MONTH_NUMBER_MAP)
    years = pd.to_numeric(extracted["year"], errors="coerce")
    parsed_dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    turkish_mask = month_numbers.notna() & years.notna()
    parsed_dates.loc[turkish_mask] = pd.to_datetime(
        {
            "year": years.loc[turkish_mask].astype(int),
            "month": month_numbers.loc[turkish_mask].astype(int),
            "day": 1,
        },
        errors="coerce",
    )

    remainder_mask = ~turkish_mask & series.notna()
    if remainder_mask.any():
        parsed_dates.loc[remainder_mask] = pd.to_datetime(series.loc[remainder_mask], errors="coerce")

    return parsed_dates


@st.cache_data(show_spinner="Veriler okunuyor ve hazırlanıyor...")
def load_and_prepare_data(adana_path: str, mersin_path: str) -> dict[str, Any]:
    adana_file = Path(adana_path)
    mersin_file = Path(mersin_path)
    if not adana_file.exists() or not mersin_file.exists():
        missing = [str(path) for path in [adana_file, mersin_file] if not path.exists()]
        raise FileNotFoundError(", ".join(missing))

    signature = cache_signature(adana_file, mersin_file)
    if PREPARED_CACHE_PATH.exists():
        try:
            cached = pd.read_pickle(PREPARED_CACHE_PATH)
            if cached.get("signature") == signature:
                return cached["payload"]
        except Exception:
            pass

    df_mersin = pd.read_excel(mersin_file, sheet_name=SHEET_NAME)
    df_adana = pd.read_excel(adana_file, sheet_name=SHEET_NAME)
    raw_df = pd.concat([df_mersin, df_adana], ignore_index=True)

    category_options = {
        "Bölge": sorted(raw_df["Bölge"].dropna().astype(str).unique().tolist()),
        "Merkez": sorted(raw_df["Merkez"].dropna().astype(str).unique().tolist()),
        "Merkez Tipi": sorted(raw_df["Merkez Tipi"].dropna().astype(str).unique().tolist()),
    }

    df = raw_df.copy()
    df["Zaman"] = parse_turkish_month_series(df["Zaman"])
    df["Anlık Kapasite Tarihi"] = pd.to_datetime(df["Anlık Kapasite Tarihi"], errors="coerce")
    r_factor = pd.to_numeric(df["R Faktörü"], errors="coerce")
    df["R Faktörü"] = r_factor.fillna(r_factor.median())

    df = _safe_drop(df, INITIAL_DROP_COLUMNS)
    df = df.dropna()

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df = _safe_drop(df, MODEL_DROP_COLUMNS)
    df = df.sort_values("Zaman")

    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(float)

    q1 = df[TARGET].quantile(0.25)
    q3 = df[TARGET].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[TARGET] >= q1 - 1.5 * iqr) & (df[TARGET] <= q3 + 1.5 * iqr)]

    df_last = df[(df["Zaman"] >= "2024-07-01") & (df["Zaman"] <= "2024-12-31")].copy()
    df_train = df[df["Zaman"] < "2024-07-01"].copy()
    df_train["Zaman"] = pd.to_datetime(df_train["Zaman"])
    df_train["Yil"] = df_train["Zaman"].dt.year
    df_train["Ay"] = df_train["Zaman"].dt.month

    feature_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in feature_columns:
        feature_columns.remove(TARGET)

    df_last_monthly = (
        df_last.set_index("Zaman")
        .resample("MS")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("Zaman")
    )

    input_stats = df_train[feature_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    latest_features = df_train[feature_columns].iloc[-1].to_dict()

    payload = {
        "df_train": df_train,
        "df_last_monthly": df_last_monthly,
        "feature_columns": feature_columns,
        "input_stats": input_stats,
        "latest_features": latest_features,
        "category_options": category_options,
        "row_counts": {
            "raw": len(raw_df),
            "train": len(df_train),
            "validation": len(df_last),
        },
    }

    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle({"signature": signature, "payload": payload}, PREPARED_CACHE_PATH)
    return payload


@st.cache_resource(show_spinner="Random Forest eğitiliyor...")
def train_random_forest(
    df_train: pd.DataFrame,
    features: list[str],
    n_estimators: int,
    random_state: int,
) -> ClassicalModel:
    x = df_train[features]
    y = df_train[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        shuffle=False,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return ClassicalModel(
        model=model,
        features=features,
        mae=mean_absolute_error(y_test, predictions),
        r2=r2_score(y_test, predictions),
    )


@st.cache_resource(show_spinner="SVM eğitiliyor...")
def train_svm(
    df_train: pd.DataFrame,
    features: list[str],
    c_value: float,
    gamma_value: float,
    epsilon_value: float,
    max_rows: int,
) -> ClassicalModel:
    training_df = df_train.tail(max_rows).copy() if len(df_train) > max_rows else df_train
    x = training_df[features]
    y = training_df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        shuffle=False,
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = SVR(kernel="rbf", C=c_value, gamma=gamma_value, epsilon=epsilon_value)
    model.fit(x_train_scaled, y_train_scaled)
    predictions_scaled = model.predict(x_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    return ClassicalModel(
        model=model,
        features=features,
        mae=mean_absolute_error(y_test, predictions),
        r2=r2_score(y_test, predictions),
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )


@st.cache_resource(show_spinner="LSTM eğitiliyor...")
def train_lstm(
    df_train: pd.DataFrame,
    features: list[str],
    epochs: int,
    batch_size: int,
    random_state: int,
) -> NeuralModel:
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
    except (ImportError, OSError) as exc:
        raise NeuralModelUnavailable(str(exc)) from exc

    tf.keras.utils.set_random_seed(random_state)
    x_raw = df_train[features]
    y_raw = df_train[TARGET]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_raw)
    y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1))
    x_3d = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))

    x_train, x_test, y_train, y_test = train_test_split(
        x_3d,
        y_scaled,
        test_size=0.2,
        shuffle=False,
    )

    model = Sequential(
        [
            Input(shape=(x_train.shape[1], x_train.shape[2])),
            LSTM(
                64,
                activation="relu",
                return_sequences=True,
            ),
            Dropout(0.2),
            LSTM(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled).ravel()
    actuals = scaler_y.inverse_transform(y_test).ravel()

    return NeuralModel(
        model=model,
        features=features,
        mae=mean_absolute_error(actuals, predictions),
        r2=r2_score(actuals, predictions),
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )


@st.cache_resource(show_spinner="CNN-GRU eğitiliyor...")
def train_cnn_gru(
    df_train: pd.DataFrame,
    features: list[str],
    epochs: int,
    batch_size: int,
    random_state: int,
) -> NeuralModel:
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import GRU, Conv1D, Dense, Dropout, Input, MaxPooling1D
        from tensorflow.keras.models import Sequential
    except (ImportError, OSError) as exc:
        raise NeuralModelUnavailable(str(exc)) from exc

    tf.keras.utils.set_random_seed(random_state)
    x_raw = df_train[features]
    y_raw = df_train[TARGET]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_raw)
    y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1))
    x_3d = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))

    x_train, x_test, y_train, y_test = train_test_split(
        x_3d,
        y_scaled,
        test_size=0.2,
        shuffle=False,
    )

    model = Sequential(
        [
            Input(shape=(x_train.shape[1], x_train.shape[2])),
            Conv1D(
                filters=64,
                kernel_size=1,
                activation="relu",
            ),
            MaxPooling1D(pool_size=1),
            GRU(48, activation="relu", return_sequences=False),
            Dropout(0.2),
            Dense(24, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled).ravel()
    actuals = scaler_y.inverse_transform(y_test).ravel()

    return NeuralModel(
        model=model,
        features=features,
        mae=mean_absolute_error(actuals, predictions),
        r2=r2_score(actuals, predictions),
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )


def predict_model(model_name: str, model_bundle: Any, rows: pd.DataFrame) -> np.ndarray:
    features = model_bundle.features
    x = rows[features]

    if model_name == "Random Forest":
        return model_bundle.model.predict(x)

    if model_name == "SVM":
        x_scaled = model_bundle.scaler_x.transform(x)
        predictions_scaled = model_bundle.model.predict(x_scaled)
        return model_bundle.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    x_scaled = model_bundle.scaler_x.transform(x)
    x_3d = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
    predictions_scaled = model_bundle.model.predict(x_3d, verbose=0)
    return model_bundle.scaler_y.inverse_transform(predictions_scaled).ravel()


def make_future_frame(features: list[str], latest_features: dict[str, float]) -> pd.DataFrame:
    future_dates = pd.date_range(start="2024-07-01", end="2024-12-31", freq="MS")
    future_df = pd.DataFrame(index=range(len(future_dates)), columns=features)
    for col in features:
        future_df[col] = latest_features.get(col, 0)
    future_df["Yil"] = future_dates.year
    future_df["Ay"] = future_dates.month
    return future_df.astype(float), future_dates


def build_forecasts(
    trained_models: dict[str, Any],
    features: list[str],
    latest_features: dict[str, float],
) -> pd.DataFrame:
    future_df, future_dates = make_future_frame(features, latest_features)
    forecast_df = pd.DataFrame({"Zaman": future_dates})
    for model_name, model_bundle in trained_models.items():
        forecast_df[model_name] = predict_model(model_name, model_bundle, future_df)
    return forecast_df


def required_model_names(selected_model: str) -> list[str]:
    return list(MODEL_COLORS) if selected_model == "Tüm Modeller" else [selected_model]


def train_selected_models(df_train: pd.DataFrame, features: list[str], model_names: list[str]) -> dict[str, Any]:
    trained_models: dict[str, Any] = {}

    if "Random Forest" in model_names:
        trained_models["Random Forest"] = train_random_forest(
            df_train,
            features,
            RF_ESTIMATORS,
            RANDOM_STATE,
        )

    if "SVM" in model_names:
        trained_models["SVM"] = train_svm(
            df_train,
            features,
            SVM_C,
            SVM_GAMMA,
            SVM_EPSILON,
            SVM_MAX_ROWS,
        )

    neural_requests = [name for name in ["LSTM", "CNN-GRU"] if name in model_names]
    if neural_requests:
        try:
            if "LSTM" in neural_requests:
                trained_models["LSTM"] = train_lstm(
                    df_train,
                    features,
                    LSTM_EPOCHS,
                    BATCH_SIZE,
                    RANDOM_STATE,
                )
            if "CNN-GRU" in neural_requests:
                trained_models["CNN-GRU"] = train_cnn_gru(
                    df_train,
                    features,
                    CNN_GRU_EPOCHS,
                    BATCH_SIZE,
                    RANDOM_STATE,
                )
        except NeuralModelUnavailable as exc:
            st.error(
                "LSTM ve CNN-GRU çalıştırılamadı. TensorFlow native DLL'i bu ortamda yüklenemiyor. "
                "Bu genelde TensorFlow/CPU uyumsuzluğu, AVX/AVX2 desteği veya Microsoft Visual C++ Redistributable "
                "eksikliğinden kaynaklanır. Random Forest ve SVM çalışmaya devam eder."
            )
            with st.expander("TensorFlow hata detayı", expanded=False):
                st.code(str(exc))
        except Exception as exc:
            st.error("LSTM/CNN-GRU eğitimi sırasında hata oluştu. Random Forest ve SVM çalışmaya devam eder.")
            with st.expander("Derin öğrenme hata detayı", expanded=False):
                st.code(repr(exc))

    return trained_models


def plot_forecast(
    selected_model: str,
    actual_monthly: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_monthly["Zaman"],
            y=actual_monthly[TARGET],
            mode="lines+markers",
            name="Gerçek Değerler",
            line={"color": "#111111", "dash": "dash", "width": 3},
            marker={"size": 8},
        )
    )

    names = list(MODEL_COLORS) if selected_model == "Tüm Modeller" else [selected_model]
    for name in names:
        if name not in forecast_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Zaman"],
                y=forecast_df[name],
                mode="lines+markers",
                name=name,
                line={"color": MODEL_COLORS[name], "dash": "dash", "width": 2.5},
                marker={"size": 8, "symbol": "square"},
            )
        )

    title = (
        "2024 İkinci Yarı Tüm Model Tahminleri"
        if selected_model == "Tüm Modeller"
        else f"2024 İkinci Yarı {selected_model} Tahmini"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Zaman",
        yaxis_title=TARGET_LABEL,
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    fig.update_xaxes(tickformat="%Y-%m", dtick="M1")
    return fig


def build_manual_input(
    features: list[str],
    stats: pd.DataFrame,
    category_options: dict[str, list[str]],
) -> pd.DataFrame:
    defaults = {feature: float(stats.loc[feature, "50%"]) for feature in features if feature in stats.index}

    with st.form("manual_prediction_form"):
        left, right = st.columns(2)
        with left:
            selected_date = st.date_input("Tahmin tarihi", value=pd.Timestamp("2024-07-01"))
            region = st.selectbox("Bölge", category_options["Bölge"])
            center = st.selectbox("Merkez", category_options["Merkez"])
            center_type = st.selectbox("Merkez Tipi", category_options["Merkez Tipi"])
        with right:
            numeric_values: dict[str, float] = {}
            for column in NUMERIC_INPUTS:
                if column not in features:
                    continue
                median = defaults.get(column, 0.0)
                q1 = float(stats.loc[column, "25%"]) if column in stats.index else median
                q3 = float(stats.loc[column, "75%"]) if column in stats.index else median
                step = max(abs(q3 - q1) / 20, 0.01)
                numeric_values[column] = st.number_input(
                    column.strip(),
                    value=median,
                    step=step,
                    format="%.6f",
                )

        submitted = st.form_submit_button("Tahmin Et")

    row = {feature: defaults.get(feature, 0.0) for feature in features}
    row.update(numeric_values)
    selected_ts = pd.Timestamp(selected_date)
    row["Yil"] = selected_ts.year
    row["Ay"] = selected_ts.month

    for feature in features:
        if feature.startswith("Bölge_"):
            row[feature] = 1.0 if feature == f"Bölge_{region}" else 0.0
        elif feature.startswith("Merkez_"):
            row[feature] = 1.0 if feature == f"Merkez_{center}" else 0.0
        elif feature.startswith("Merkez Tipi_"):
            row[feature] = 1.0 if feature == f"Merkez Tipi_{center_type}" else 0.0

    return pd.DataFrame([row], columns=features).astype(float), submitted


def metric_table(trained_models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, model_bundle in trained_models.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": model_bundle.mae,
                "R2": model_bundle.r2,
                "Özellik Sayısı": len(model_bundle.features),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Forecasting", layout="wide")
    st.title("Forecasting")

    with st.sidebar:
        selected_model = st.radio("Grafik", MODEL_OPTIONS)

    try:
        adana_path, mersin_path = resolve_data_paths()
        prepared = load_and_prepare_data(adana_path, mersin_path)
    except FileNotFoundError as exc:
        st.error(f"Dosya bulunamadı: {exc}")
        st.stop()

    df_train = prepared["df_train"]
    features = prepared["feature_columns"]
    row_counts = prepared["row_counts"]

    chart_model_names = required_model_names(selected_model)
    trained_models = train_selected_models(df_train, features, chart_model_names)

    if not trained_models:
        st.warning(
            "Seçilen grafik için model üretilemedi. TensorFlow sorunu çözülene kadar Random Forest veya SVM grafiğini kullanabilirsin."
        )

    forecast_df = build_forecasts(
        trained_models,
        features,
        prepared["latest_features"],
    )

    count_cols = st.columns(3)
    count_cols[0].metric("Ham satır", f"{row_counts['raw']:,}")
    count_cols[1].metric("Eğitim satırı", f"{row_counts['train']:,}")
    count_cols[2].metric("2024 test satırı", f"{row_counts['validation']:,}")

    chart_model = selected_model
    if chart_model not in trained_models and chart_model != "Tüm Modeller":
        chart_model = "Tüm Modeller"

    st.plotly_chart(
        plot_forecast(chart_model, prepared["df_last_monthly"], forecast_df),
        use_container_width=True,
    )

    with st.expander("Model performansları", expanded=True):
        st.dataframe(
            metric_table(trained_models),
            hide_index=True,
            use_container_width=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.4f"),
                "R2": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    with st.expander("Forecasting değerleri", expanded=False):
        st.dataframe(
            forecast_df,
            hide_index=True,
            use_container_width=True,
            column_config={"Zaman": st.column_config.DateColumn(format="YYYY-MM")},
        )

    st.subheader("Feature Girişleri ile Target Tahmini")
    manual_frame, submitted = build_manual_input(
        features,
        prepared["input_stats"],
        prepared["category_options"],
    )

    if submitted:
        manual_models = train_selected_models(df_train, features, list(MODEL_COLORS))
        if not manual_models:
            st.warning("Tahmin üretilemedi; çalışan en az bir model gerekli.")
            st.stop()
        manual_rows = []
        for model_name, model_bundle in manual_models.items():
            manual_rows.append(
                {
                    "Model": model_name,
                    TARGET: float(predict_model(model_name, model_bundle, manual_frame)[0]),
                }
            )
        st.dataframe(
            pd.DataFrame(manual_rows),
            hide_index=True,
            use_container_width=True,
            column_config={TARGET: st.column_config.NumberColumn(format="%.4f")},
        )


if __name__ == "__main__":
    main()
