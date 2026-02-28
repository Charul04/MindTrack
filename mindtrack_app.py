import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="MindTrack AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #080b12 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }

/* Main content padding */
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 1400px; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f1923 0%, #111827 50%, #0a0f1a 100%);
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-title span { color: #63b3ed; }
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.2px;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #2d4a6b; transform: translateY(-2px); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #63b3ed); }
.metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.metric-card.red::before   { background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-label {
    font-size: 0.75rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2535;
    margin-left: 0.5rem;
}

/* Chart containers */
.chart-card {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 1.5rem;
    height: 100%;
}
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
}

/* Prediction result */
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 1px solid;
    margin: 1.5rem 0;
}
.result-box.low    { background: #052e16; border-color: #166534; }
.result-box.medium { background: #1c1008; border-color: #92400e; }
.result-box.high   { background: #1c0a0a; border-color: #991b1b; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-box.low    .result-label { color: #4ade80; }
.result-box.medium .result-label { color: #fb923c; }
.result-box.high   .result-label { color: #f87171; }
.result-desc { color: #94a3b8; font-size: 0.95rem; }

/* Prob bars */
.prob-row { margin: 0.4rem 0; }
.prob-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.2rem; display:flex; justify-content:space-between; }
.prob-bar-bg { background: #1e2535; border-radius: 999px; height: 8px; }
.prob-bar-fill { height: 8px; border-radius: 999px; transition: width 0.6s ease; }

/* Input form styling */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stSlider"] {
    background: #0d1117 !important;
    border-color: #1e2535 !important;
    color: #e2e8f0 !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* Divider */
hr { border-color: #1e2535 !important; }

/* Stmetric override */
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'Syne', sans-serif !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #1e2535 !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#1e2535",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#94a3b8",
    "grid.color":        "#1e2535",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = ["#3b82f6", "#f59e0b", "#ef4444"]
LABELS = ["Low Risk", "At-Risk", "High Risk"]

@st.cache_data
def load_data():
    return pd.read_csv("mindtrack_dataset_final.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load("mindtrack_model.pkl")
    except:
        df = load_data()
        X = df.drop(columns=["Risk_Level"])
        y = df["Risk_Level"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.07,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="mlogloss", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        joblib.dump(model, "mindtrack_model.pkl")
        return model

df      = load_data()
model   = load_model()
X_all   = df.drop(columns=["Risk_Level"])
y_all   = df["Risk_Level"]
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)


with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#f1f5f9'>
            🧠 MindTrack
        </div>
        <div style='font-size:0.78rem;color:#334155;margin-top:0.2rem;letter-spacing:0.5px'>
            STUDENT MENTAL HEALTH AI
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["Overview", "Model Performance", "Predictor"], label_visibility="collapsed")
    st.divider()
    total   = len(df)
    at_risk = (df["Risk_Level"] >= 1).sum()
    st.markdown(f"""
    <div style='padding:1rem;background:#0d1117;border:1px solid #1e2535;border-radius:10px;margin-top:0.5rem'>
        <div style='font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem'>DATASET INFO</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'>
            <span style='color:#475569;font-size:0.82rem'>Total Students</span>
            <span style='color:#e2e8f0;font-weight:600;font-size:0.82rem'>{total:,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'>
            <span style='color:#475569;font-size:0.82rem'>At Risk / High</span>
            <span style='color:#f59e0b;font-weight:600;font-size:0.82rem'>{at_risk:,}</span>
        </div>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#475569;font-size:0.82rem'>Features</span>
            <span style='color:#e2e8f0;font-weight:600;font-size:0.82rem'>{X_all.shape[1]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if page == "Overview":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>Student Mental Health<br><span>Early Warning System</span></div>
        <div class='hero-subtitle'>XGBoost-powered risk detection across 27,000+ student records</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    low_n  = (df["Risk_Level"] == 0).sum()
    mid_n  = (df["Risk_Level"] == 1).sum()
    high_n = (df["Risk_Level"] == 2).sum()
    acc    = (y_pred == y_test.values).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card blue'>
            <div class='metric-label'>Total Records</div>
            <div class='metric-value'>{total:,}</div>
            <div class='metric-sub'>Students surveyed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card green'>
            <div class='metric-label'>Low Risk</div>
            <div class='metric-value'>{low_n:,}</div>
            <div class='metric-sub'>{low_n/total*100:.1f}% of students</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card amber'>
            <div class='metric-label'>At-Risk</div>
            <div class='metric-value'>{mid_n:,}</div>
            <div class='metric-sub'>{mid_n/total*100:.1f}% of students</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>High Risk</div>
            <div class='metric-value'>{high_n:,}</div>
            <div class='metric-sub'>{high_n/total*100:.1f}% of students</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        counts  = df["Risk_Level"].value_counts().sort_index()
        bars    = ax.bar(LABELS, counts.values, color=COLORS, width=0.5, zorder=3)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f"{val:,}", ha="center", fontsize=10, color="#94a3b8", fontweight="600")
        ax.set_ylabel("Students", fontsize=10)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>Risk Breakdown</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.8), subplot_kw=dict(aspect="equal"))
        wedge_colors = ["#1d4ed8", "#d97706", "#dc2626"]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=LABELS,
            colors=wedge_colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2)
        )
        for t in texts:      t.set_color("#64748b"); t.set_fontsize(9)
        for t in autotexts:  t.set_color("#f1f5f9"); t.set_fontsize(9); t.set_fontweight("bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    
    st.markdown("<div class='section-header'>Key Indicators Overview</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        dep_rate = df.groupby("Risk_Level")["Depression"].mean() * 100
        ax.bar(LABELS, dep_rate.values, color=COLORS, width=0.5, zorder=3)
        ax.set_title("Depression Rate by Risk Level", fontsize=10, color="#64748b")
        ax.set_ylabel("% with Depression", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sleep_avg = df.groupby("Risk_Level")["Sleep_Quality"].mean()
        ax.bar(LABELS, sleep_avg.values, color=COLORS, width=0.5, zorder=3)
        ax.set_title("Avg Sleep Quality by Risk Level", fontsize=10, color="#64748b")
        ax.set_ylabel("Sleep Score (1-4)", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col5:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        stress_avg = df.groupby("Risk_Level")["Financial_Stress"].mean()
        ax.bar(LABELS, stress_avg.values, color=COLORS, width=0.5, zorder=3)
        ax.set_title("Avg Financial Stress by Risk Level", fontsize=10, color="#64748b")
        ax.set_ylabel("Stress Score (1-5)", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()



elif page == "Model Performance":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.3rem'>
        Model Performance
    </div>
    <div style='color:#475569;font-size:0.95rem;margin-bottom:2rem'>
        Performance metrics and feature analysis
    </div>
    """, unsafe_allow_html=True)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="weighted")
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc   = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card blue'>
            <div class='metric-label'>Accuracy</div>
            <div class='metric-value'>{acc*100:.1f}%</div>
            <div class='metric-sub'>Test set</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card green'>
            <div class='metric-label'>F1 Score</div>
            <div class='metric-value'>{f1:.3f}</div>
            <div class='metric-sub'>Weighted avg</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card amber'>
            <div class='metric-label'>ROC-AUC</div>
            <div class='metric-value'>{roc:.3f}</div>
            <div class='metric-sub'>Macro OvR</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card red'>
            <div class='metric-label'>Test Samples</div>
            <div class='metric-value'>{len(y_test):,}</div>
            <div class='metric-sub'>20% holdout</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Feature Importance  ", "  Confusion Matrix  ", "  ROC Curves  "])

    with tab1:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
            importance = pd.Series(model.feature_importances_, index=X_all.columns).sort_values()
            fig, ax    = plt.subplots(figsize=(8, 5))
            colors_imp = ["#3b82f6" if v == importance.max() else "#1e3a5f" for v in importance.values]
            bars = ax.barh(importance.index, importance.values, color=colors_imp, height=0.6, zorder=3)
            for bar, val in zip(bars, importance.values):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=8.5, color="#64748b")
            ax.set_xlabel("Importance Score")
            ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("<div class='section-header'>Top Features</div>", unsafe_allow_html=True)
            top5 = importance.nlargest(5)
            for feat, val in top5.items():
                pct = int(val / importance.max() * 100)
                st.markdown(f"""
                <div style='margin-bottom:1rem'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                        <span style='color:#94a3b8;font-size:0.85rem'>{feat.replace("_"," ")}</span>
                        <span style='color:#3b82f6;font-size:0.85rem;font-weight:600'>{val:.3f}</span>
                    </div>
                    <div style='background:#1e2535;border-radius:99px;height:6px'>
                        <div style='background:linear-gradient(90deg,#1d4ed8,#3b82f6);width:{pct}%;height:6px;border-radius:99px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
            cm  = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=LABELS, yticklabels=LABELS,
                        linewidths=1.5, linecolor="#0d1117",
                        cbar_kws={"shrink": 0.8}, ax=ax,
                        annot_kws={"size": 13, "weight": "bold", "color": "#f1f5f9"})
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.tick_params(colors="#64748b")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("<div class='section-header'>Per-Class Accuracy</div>", unsafe_allow_html=True)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            fig, ax = plt.subplots(figsize=(5, 5))
            for i, (label, color) in enumerate(zip(LABELS, COLORS)):
                ax.barh(label, cm_norm[i, i], color=color, height=0.4, zorder=3)
                ax.text(cm_norm[i, i] + 0.5, i, f"{cm_norm[i,i]:.1f}%",
                        va="center", fontsize=11, color="#94a3b8", fontweight="600")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Recall (%)")
            ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("<div class='section-header'>ROC Curves — One vs Rest</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.3, 1])
        with col1:
            y_bin = label_binarize(y_test, classes=[0, 1, 2])
            fig, ax = plt.subplots(figsize=(7, 5))
            for i in range(3):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                score       = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=COLORS[i], lw=2.5,
                        label=f"{LABELS[i]}  (AUC = {score:.3f})")
            ax.fill_between([0,1],[0,1], alpha=0.04, color="white")
            ax.plot([0,1],[0,1],"--", color="#334155", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(framealpha=0.1, loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("<div class='section-header'>AUC Summary</div>", unsafe_allow_html=True)
            for i in range(3):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                score = auc(fpr, tpr)
                pct   = int(score * 100)
                st.markdown(f"""
                <div style='margin-bottom:1.2rem'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                        <span style='color:#94a3b8;font-size:0.88rem'>{LABELS[i]}</span>
                        <span style='color:{COLORS[i]};font-size:0.88rem;font-weight:700'>{score:.3f}</span>
                    </div>
                    <div style='background:#1e2535;border-radius:99px;height:7px'>
                        <div style='background:{COLORS[i]};width:{pct}%;height:7px;border-radius:99px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)



elif page == "Predictor":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.3rem'>
        Live Risk Predictor
    </div>
    <div style='color:#475569;font-size:0.95rem;margin-bottom:2rem'>
        Enter student details to get an instant mental health risk assessment
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Student Profile</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1: age    = st.slider("Age", 17, 35, 21)
        with c2: gender = st.selectbox("Gender", ["Male (0)", "Female (1)"])
        gender_val = 0 if "Male" in gender else 1

        c3, c4 = st.columns(2)
        with c3: cgpa      = st.slider("CGPA (0–10)", 0.0, 10.0, 7.5, 0.1)
        with c4: acad_pres = st.slider("Academic Pressure (1–5)", 1, 5, 3)

        c5, c6 = st.columns(2)
        with c5: study_sat  = st.slider("Study Satisfaction (1–5)", 1, 5, 3)
        with c6: sleep      = st.selectbox("Sleep Quality", ["< 5 hrs (1)", "5–6 hrs (2)", "7–8 hrs (3)", "> 8 hrs (4)"])
        sleep_val = int(sleep.split("(")[1].replace(")", ""))

        c7, c8 = st.columns(2)
        with c7: diet = st.selectbox("Dietary Habits", ["Unhealthy (1)", "Moderate (2)", "Healthy (3)"])
        with c8: study_hrs = st.slider("Study Hours/Day", 0.0, 12.0, 6.0, 0.5)
        diet_val = int(diet.split("(")[1].replace(")", ""))

        c9, c10 = st.columns(2)
        with c9:  fin_stress  = st.slider("Financial Stress (1–5)", 1, 5, 3)
        with c10: suicidal    = st.selectbox("Suicidal Thoughts", ["No (0)", "Yes (1)"])
        suicidal_val = int(suicidal.split("(")[1].replace(")", ""))

        c11, c12 = st.columns(2)
        with c11: family_hist = st.selectbox("Family Mental Health History", ["No (0)", "Yes (1)"])
        with c12: depression  = st.selectbox("Depression Diagnosed", ["No (0)", "Yes (1)"])
        family_val     = int(family_hist.split("(")[1].replace(")", ""))
        depression_val = int(depression.split("(")[1].replace(")", ""))

        predict_btn = st.button("🔍  Run Assessment", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'>Risk Assessment</div>", unsafe_allow_html=True)

        if predict_btn:
            input_data = pd.DataFrame([{
                "Age":                age,
                "Gender":             gender_val,
                "CGPA":               cgpa,
                "Academic_Pressure":  acad_pres,
                "Study_Satisfaction": study_sat,
                "Sleep_Quality":      sleep_val,
                "Dietary_Habits":     diet_val,
                "Suicidal_Thoughts":  suicidal_val,
                "Study_Hours":        study_hrs,
                "Financial_Stress":   fin_stress,
                "Family_History":     family_val,
                "Depression":         depression_val
            }])

            proba      = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]

            box_class = ["low", "medium", "high"][prediction]
            result_labels = ["🟢 Low Risk", "🟡 At-Risk", "🔴 High Risk"]
            descriptions  = [
                "Student appears mentally healthy. Maintain current habits.",
                "Moderate risk detected. Consider counseling check-in.",
                "High risk indicators present. Immediate support recommended."
            ]

            st.markdown(f"""
            <div class='result-box {box_class}'>
                <div class='result-label'>{result_labels[prediction]}</div>
                <div class='result-desc'>{descriptions[prediction]}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Probability Breakdown</div>", unsafe_allow_html=True)

            bar_colors = ["#3b82f6", "#f59e0b", "#ef4444"]
            for i, (label, prob, color) in enumerate(zip(LABELS, proba, bar_colors)):
                pct = int(prob * 100)
                st.markdown(f"""
                <div class='prob-row'>
                    <div class='prob-label'>
                        <span>{label}</span>
                        <span style='color:{color};font-weight:700'>{prob*100:.1f}%</span>
                    </div>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill' style='width:{pct}%;background:{color}'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:0.6rem'></div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header' style='margin-top:1.5rem'>Input Profile</div>", unsafe_allow_html=True)
            features_show = ["Academic_Pressure", "Financial_Stress", "Sleep_Quality",
                             "Study_Satisfaction", "Dietary_Habits"]
            values_show   = [acad_pres, fin_stress, sleep_val, study_sat, diet_val]
            maxvals       = [5, 5, 4, 5, 3]
            norm_vals     = [v/m for v, m in zip(values_show, maxvals)]

            fig, ax = plt.subplots(figsize=(6, 2.8))
            bar_colors_profile = [COLORS[prediction]] * len(features_show)
            bars = ax.barh([f.replace("_", " ") for f in features_show],
                           norm_vals, color=bar_colors_profile, height=0.5, zorder=3)
            for bar, val, raw in zip(bars, norm_vals, values_show):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        str(raw), va="center", fontsize=9, color="#64748b")
            ax.set_xlim(0, 1.2)
            ax.set_xlabel("Normalized Score")
            ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;background:#0d1117;
                        border:1px dashed #1e2535;border-radius:14px;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem'>🔍</div>
                <div style='color:#334155;font-size:0.95rem'>
                    Fill in the student profile and<br>click <b style='color:#475569'>Run Assessment</b>
                </div>
            </div>""", unsafe_allow_html=True)
