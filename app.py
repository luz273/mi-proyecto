import streamlit as st    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import seaborn as sns
import io

# Configuración inicial de la página
st.set_page_config(page_title="Predicción de Defectos", page_icon="🔍", layout="wide")

# Estilo personalizado
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .classification-report {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción con mejor formato
st.markdown("# 🔍 Predicción de Defectos en Productos o Servicios")
st.markdown("### Análisis predictivo de defectos basado en datos históricos")

# Variables predefinidas
TARGET_COLUMN = "Defectuoso"
FEATURE_COLUMNS = ["Productos-Lote", "Tiempo-Entrega"]

# Sidebar con mejor organización
with st.sidebar:
    st.markdown("### 🛠️ Configuración")
    uploaded_file = st.file_uploader("📂 Cargar datos (CSV o Excel)", type=["csv", "xlsx"])

# Función para cargar datos desde distintos formatos
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Formato de archivo no soportado. Sube un archivo CSV o Excel.")
        return None

def create_distribution_plot(y):
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="viridis", ax=ax)
    ax.set_title("Distribución de la Variable Objetivo")
    return fig

def create_feature_distributions(df_clean, FEATURE_COLUMNS):
    fig, axes = plt.subplots(1, len(FEATURE_COLUMNS), figsize=(15, 5))
    if len(FEATURE_COLUMNS) == 1:
        axes = [axes]
    for i, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df_clean[col], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribución de {col}")
    plt.tight_layout()
    return fig

def create_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matriz de Confusión")
    return fig

def create_decision_boundary(X_scaled, y, model, xx, yy, Z):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors="k", cmap="coolwarm", s=50)
    ax.set_xlabel(FEATURE_COLUMNS[0])
    ax.set_ylabel(FEATURE_COLUMNS[1])
    ax.set_title("Frontera de Decisión")
    plt.colorbar(scatter)
    return fig

def create_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    return fig

def create_pr_curve(precision, recall, avg_precision):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue", label=f'AP = {avg_precision:.2f}')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    return fig

# Comprobar si se ha subido un archivo
if uploaded_file:
    try:
        # Cargar datos
        df = load_data(uploaded_file)
        if df is not None:
            # Crear columnas para mejor organización
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📋 Vista Previa de los Datos")
                st.dataframe(df.head(), use_container_width=True)

            with col2:
                st.markdown("### 📊 Información del Modelo")
                st.write(f"**Variable objetivo:** {TARGET_COLUMN}")
                st.write(f"**Características:** {', '.join(FEATURE_COLUMNS)}")

            # Preparación de datos
            df_clean = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
            X = df_clean[FEATURE_COLUMNS]
            y = df_clean[TARGET_COLUMN]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # División de datos y entrenamiento
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
            param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
            grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            # Predicciones y métricas
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            # Frontera de decisión
            xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                                np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            # Cálculo de métricas adicionales
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            avg_precision = average_precision_score(y_test, y_pred_prob)
            f1 = f1_score(y_test, y_pred)

            # Modelo de Regresión Logística
            st.markdown("### 📊 Modelo de Regresión Logística")
            
            # Coeficientes
            coef_df = pd.DataFrame({
                'Característica': FEATURE_COLUMNS,
                'Coeficiente': model.coef_[0]
            })
            st.dataframe(coef_df, use_container_width=True)

            # Ecuación del modelo
            st.markdown("#### Ecuación del Modelo:")
            equation = f"log(p/(1-p)) = {model.intercept_[0]:.3f}"
            for coef, feature in zip(model.coef_[0], FEATURE_COLUMNS):
                equation += f" + ({coef:.3f} × {feature})"
            st.markdown(f"```{equation}```")

            # Métricas en tarjetas
            st.markdown("### 📈 Métricas del Modelo")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precisión CV", f"{cross_val_score(model, X_scaled, y, cv=5).mean():.2f}")
                st.metric("F1-Score", f"{f1:.2f}")
            
            with col2:
                st.metric("AUC-ROC", f"{roc_auc:.2f}")
                st.metric("Avg Precision", f"{avg_precision:.2f}")
            
            with col3:
                st.metric("Mejor Parámetro C", f"{model.get_params()['C']:.2f}")
                st.metric("Recall", f"{classification_rep['1']['recall']:.2f}")
                
            with col4:
                st.metric("Exactitud", f"{model.score(X_test, y_test):.2f}")
                st.metric("Precision", f"{classification_rep['1']['precision']:.2f}")

            # Reporte de Clasificación Detallado
            st.markdown("### 📊 Reporte de Clasificación Detallado")
            
            # Convertir el reporte de clasificación a DataFrame para mejor visualización
            report_df = pd.DataFrame(classification_rep).transpose()
            # Eliminar las columnas que no queremos mostrar
            if 'support' in report_df.columns:
                report_df = report_df.drop('support', axis=1)
            
            # Mostrar el reporte en una tabla estilizada
            st.dataframe(
                report_df.style.format("{:.2f}")
                .background_gradient(cmap='Blues'),
                use_container_width=True
            )

            # Visualizaciones
            st.markdown("### 📊 Visualizaciones")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("📊 Distribución"):
                    st.pyplot(create_distribution_plot(y))
                    st.pyplot(create_feature_distributions(df_clean, FEATURE_COLUMNS))
                
            with col2:
                if st.button("🎯 Matriz Confusión"):
                    st.pyplot(create_confusion_matrix(cm))
                
            with col3:
                if st.button("📈 Curva ROC"):
                    st.pyplot(create_roc_curve(fpr, tpr, roc_auc))
                    
            with col4:
                if st.button("🎲 Frontera"):
                    st.pyplot(create_decision_boundary(X_scaled, y, model, xx, yy, Z))

            with col5:
                if st.button("📉 Precision-Recall"):
                    st.pyplot(create_pr_curve(precision, recall, avg_precision))

            # Predicción con nuevos datos
            st.markdown("### 🎯 Realizar Predicción")
            col1, col2 = st.columns(2)
            new_data = {}
            
            with col1:
                for col in FEATURE_COLUMNS:
                    new_data[col] = st.number_input(f'Valor para {col}', value=0.0)
            
            with col2:
                new_data_df = pd.DataFrame([new_data])
                new_data_scaled = scaler.transform(new_data_df)
                prediction = model.predict(new_data_scaled)
                prob = model.predict_proba(new_data_scaled)[0][1]
                
                st.markdown("#### Resultado:")
                if prediction[0] == 1:
                    st.error(f"⚠️ Defectuoso (Probabilidad: {prob:.2%})")
                else:
                    st.success(f"✅ No Defectuoso (Probabilidad: {1-prob:.2%})")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("👆 Sube un archivo CSV o Excel en el panel lateral para comenzar.")