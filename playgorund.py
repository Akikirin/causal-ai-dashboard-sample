import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import warnings
from scipy.stats import norm

# --- 1. Machine Learning & Causal Imports ---
try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("🚨 Please install xgboost: `pip install xgboost`")
    st.stop()

try:
    from econml.dml import LinearDML
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install econml`")
    st.stop()

try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install causalml`")
    st.stop()

from fpdf import FPDF

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 2. Page Configuration & Professional CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Universal Causal Dashboard", page_icon="🔮")

# Translation Dictionary
LANG_DICT = {
    "title": "🧠 Causal AI Strategy Dashboard / 因果人工智慧策略儀表板",
    "subtitle": "Quantify the **True Impact** of your decisions / 使用雙重機器學習 (DML) 量化決策的**真實影響力**",
    "control_tower": "🎛️ Control Tower / 控制塔",
    "upload_info": "Upload historical sales data / 請上傳歷史銷售數據以進行因果推斷",
    "upload_label": "Upload CSV Data / 上傳 CSV 資料",
    "conf_1": "### 1. Model Configuration / 模型配置",
    "target_y": "🎯 Target (Outcome Y) / 目標變數 (結果 Y)",
    "treatment_t": "💊 Treatment (Input T) / 干預變數 (投入 T)",
    "confounders_w": "🌪️ Confounders (Controls W) / 混雜變數 (控制變數 W)",
    "conf_2": "### 2. Execution / 執行",
    "run_btn": "🚀 Run Causal Engine / 啟動因果引擎",
    "obs_window": "Observation Window / 觀察週期",
    "track_conf": "Confounders Tracked / 已追蹤混雜變數",
    "tab1": "⚡ Insights & Elasticity / 洞察與彈性",
    "tab2": "🔮 Simulator / 模擬器",
    "tab3": "⚔️ Model Battle / 模型競技場",
    "tab4": "⚖️ Evaluation / 模型評估",
    "tab5": "🌌 Parallel Universe / 平行宇宙",
    "tab6": "📋 Executive Report / 執行報告",
    "noise_cancel": "Separating Signal from Noise / 過濾噪聲提取訊號",
    "theory_logic": "<b>Noise Cancellation Logic / 降噪邏輯:</b><br>Standard correlations are biased because <b>Confounders</b> affect both T and Y. We use <b>DML</b> to isolate the pure causal link.<br>標準相關性通常存在偏差，因為<b>混雜變數</b>同時影響決策與結果。我們使用<b>雙重機器學習 (DML)</b> 來隔離出純粹的因果關係。",
    "true_elasticity": "True Causal Elasticity / 真實因果彈性",
    "naive_corr": "Naive Correlation / 原始相關性",
    "bias_detected": "Bias Detected / 偵測到偏差",
    "data_reliability": "Data Reliability / 數據可靠性",
    "market_momentum": "Market Momentum (PCA) / 市場動能",
    "causal_impact": "Causal Impact / 因果影響",
    "scenario_breakdown": "📊 Scenario Breakdown / 情境分析細節",
    "u_a": "Universe A (Status Quo) / 宇宙 A (現狀)",
    "u_b": "Universe B (Causal AI) / 宇宙 B (因果 AI)",
    "rev_comp": "💰 Total Revenue / 總營收",
    "inv_comp": "📦 Avg Inventory / 平均庫存",
    "sl_comp": "🤝 Service Level / 服務水準",
    "short_comp": "🚫 Shortage Events / 缺貨事件",
    "improvement": "Improvement / 改善幅度"
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans&display=swap');
    html, body, [class*="css"], font, div, span, p, text { font-family: 'Josefin Sans', sans-serif !important; }
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-weight: 800 !important; }
    .theory-box { background-color: #f0f9ff; border-left: 5px solid #0ea5e9; padding: 15px; border-radius: 4px; color: #334155; }
    .comp-delta-pos { color: #10b981; font-weight: bold; }
    .comp-delta-neg { color: #ef4444; font-weight: bold; }
    .comp-label { font-weight: bold; color: #1e293b; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Data Processing & Logic
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def auto_feature_eng(df, target, treatment):
    df = df.copy()
    lag_cols = []
    for i in [1, 2, 3]:
        col_name = f'{target}_Lag{i}'
        df[col_name] = df[target].shift(i)
        lag_cols.append(col_name)
    df = df.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    if lag_cols:
        pca = PCA(n_components=1)
        df['Latent_Market_State'] = pca.fit_transform(scaler.fit_transform(df[lag_cols]))
    else:
        df['Latent_Market_State'] = 0
    return df

def simulate_inventory_dynamic(demand_series, target_series, lead_time):
    inv_levels = []
    current_stock = target_series[0]
    shortage_events = 0
    lost_sales = []
    for day in range(len(demand_series)):
        demand = demand_series[day]
        if current_stock >= demand:
            current_stock -= demand
            lost_sales.append(0)
        else:
            lost = demand - current_stock
            lost_sales.append(lost)
            current_stock = 0
            shortage_events += 1
        inv_levels.append(current_stock)
        if (day + 1) % lead_time == 0:
            next_target = target_series[min(day+1, len(target_series)-1)]
            current_stock = next_target 
    return np.array(inv_levels), shortage_events, np.array(lost_sales)

# ==========================================
# 4. Engine Classes
# ==========================================
class RealCausalEngine:
    def __init__(self):
        self.dml_est = LinearDML(
            model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5),
            model_t=RandomForestRegressor(n_estimators=50, min_samples_leaf=5),
            random_state=42, cv=3
        )
        self.base_model = XGBRegressor(n_estimators=100, random_state=42)
        self.features = []; self.treatment = ""; self.confounders = []

    def train(self, df, target_col, treatment_col, confounders, heterogeneity_cols=None):
        self.target = target_col; self.treatment = treatment_col; self.confounders = confounders
        X = df[heterogeneity_cols] if heterogeneity_cols else None
        W = df[confounders]; Y = df[target_col]; T = df[treatment_col]
        with st.spinner("🧠 Engines warming up... DML running 3-Fold Cross-Fitting..."):
            self.dml_est.fit(Y, T, X=X, W=W)
            all_feats = [treatment_col] + confounders + (heterogeneity_cols if heterogeneity_cols else [])
            self.base_model.fit(df[all_feats], Y)
            self.features = all_feats

    def get_causal_effect(self, X_pred):
        return self.dml_est.effect(X_pred)

    def predict_counterfactual(self, df_input, new_price_col):
        base_pred = self.base_model.predict(df_input[self.features])
        delta_t = df_input[new_price_col] - df_input[self.treatment]
        theta = self.dml_est.effect(df_input[['Latent_Market_State']]) if 'Latent_Market_State' in df_input.columns else self.dml_est.const_marginal_effect(df_input[self.confounders])
        return np.maximum(base_pred + (theta * delta_t), 0)

def train_meta_learners(df, target_col, treatment_col, feature_cols):
    X = df[feature_cols]; y = df[target_col]; w = df[treatment_col].copy()
    w_binary = (w > w.median()).astype(int) if w.nunique() > 2 else w.astype(int)
    results = {}
    results['S-Learner'] = BaseSRegressor(learner=LinearRegression()).fit_predict(X=X, treatment=w_binary, y=y).flatten()
    results['T-Learner'] = BaseTRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0)).fit_predict(X=X, treatment=w_binary, y=y).flatten()
    results['X-Learner'] = BaseXRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0)).fit_predict(X=X, treatment=w_binary, y=y).flatten()
    return pd.DataFrame(results)

# ==========================================
# 5. Main Application Logic
# ==========================================
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title(LANG_DICT["title"])
    st.markdown(LANG_DICT["subtitle"])

with st.sidebar:
    st.header(LANG_DICT["control_tower"])
    st.info(LANG_DICT["upload_info"])
    uploaded_file = st.file_uploader(LANG_DICT["upload_label"], type="csv")
    
    if uploaded_file:
        raw_df = load_data(uploaded_file)
        cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        st.markdown(LANG_DICT["conf_1"])
        target_col = st.selectbox(LANG_DICT["target_y"], cols, index=0)
        treatment_col = st.selectbox(LANG_DICT["treatment_t"], cols, index=1)
        avail_cols = [c for c in cols if c not in [target_col, treatment_col]]
        confounders = st.multiselect(LANG_DICT["confounders_w"], avail_cols, default=avail_cols[:2])
        st.markdown(LANG_DICT["conf_2"])
        if st.button(LANG_DICT["run_btn"], type="primary", use_container_width=True):
            st.session_state['run'] = True
            st.session_state['cate_results'] = None; st.session_state['fold_metrics'] = None
    else:
        st.caption("Waiting for data... / 等待數據上傳...")

if st.session_state.get('run', False) and uploaded_file:
    df_eng = auto_feature_eng(raw_df, target_col, treatment_col)
    train_size = int(len(df_eng) * 0.8)
    train_df = df_eng.iloc[:train_size]
    test_df = df_eng.iloc[train_size:].reset_index(drop=True)
    all_confounders = confounders + [c for c in df_eng.columns if 'Lag' in c]

    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(LANG_DICT["obs_window"], f"{len(df_eng)} P")
    m_col2.metric("Target / 目標", target_col)
    m_col3.metric("Treatment / 干預", treatment_col)
    m_col4.metric(LANG_DICT["track_conf"], len(all_confounders))
    st.markdown("---")

    if 'main_engine' not in st.session_state:
        engine = RealCausalEngine()
        engine.train(train_df, target_col, treatment_col, all_confounders, heterogeneity_cols=['Latent_Market_State'])
        st.session_state['main_engine'] = engine
    else:
        engine = st.session_state['main_engine']

    effects = engine.get_causal_effect(test_df[['Latent_Market_State']])
    avg_elasticity = np.mean(effects)
    naive_corr = test_df[[treatment_col, target_col]].corr().iloc[0,1]
    bias_delta = avg_elasticity - naive_corr

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([LANG_DICT[f"tab{i}"] for i in range(1, 7)])
    
    with tab1:
        st.subheader(LANG_DICT["noise_cancel"])
        c_dag, c_explain = st.columns([1, 2])
        with c_dag:
            dot = graphviz.Digraph(); dot.attr(rankdir='LR')
            dot.node('T', 'Treatment / 干預', fillcolor='#d1fae5', style='filled') 
            dot.node('Y', 'Outcome / 結果', fillcolor='#dbeafe', style='filled')
            dot.node('W', 'Confounders / 混雜', shape='ellipse', fillcolor='#fee2e2', style='filled')
            dot.edge('T', 'Y', label=' Causal Link'); dot.edge('W', 'T', style='dashed'); dot.edge('W', 'Y', style='dashed')
            st.graphviz_chart(dot)
        with c_explain:
            st.markdown(f"<div class='theory-box'>{LANG_DICT['theory_logic']}</div>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric(LANG_DICT["true_elasticity"], f"{avg_elasticity:.3f}")
        k2.metric(LANG_DICT["naive_corr"], f"{naive_corr:.3f}", delta=f"{bias_delta:.3f}", delta_color="inverse")
        k3.metric(LANG_DICT["data_reliability"], "Significant Bias" if abs(bias_delta) > 0.1 else "Clean Data")

        fig_hte = px.scatter(x=test_df['Latent_Market_State'], y=effects, color=effects, template="plotly_white", 
                             labels={'x': LANG_DICT["market_momentum"], 'y': LANG_DICT["causal_impact"]})
        st.plotly_chart(fig_hte, use_container_width=True)

    with tab2:
        st.subheader("🔮 Simulator / 模擬器")
        col_in, col_out = st.columns([1, 2])
        with col_in:
            price_main = st.slider("Proposed Value / 建議數值", float(test_df[treatment_col].min()), float(test_df[treatment_col].max()), float(test_df[treatment_col].mean()))
            lead_time = st.number_input("Lead Time (Days) / 前置時間", value=5)
        with col_out:
            sim_df = test_df.copy(); sim_df[f'New_{treatment_col}'] = price_main
            cf_main = engine.predict_counterfactual(sim_df, f'New_{treatment_col}')
            st.metric("Projected Demand / 預測需求", f"{cf_main.sum():,.0f}")
            fig_cf = px.line(y=cf_main, title="Counterfactual Forecast / 反事實預測", template="plotly_white")
            st.plotly_chart(fig_cf, use_container_width=True)

    with tab5:
        st.subheader(LANG_DICT["tab5"])
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            price_b = st.slider("Universe B Price / 宇宙 B 價格", float(test_df[treatment_col].min()), float(test_df[treatment_col].max()), float(test_df[treatment_col].mean()))
            target_sl = st.slider("Target Service Level / 目標服務水準 (%)", 90, 99, 95) / 100
            sim_lt = st.number_input("Lead Time / 前置時間", value=5, key="p5_lt")
        
        with col_p2:
            # Universe A logic
            d_a = test_df[target_col]; p_a = float(test_df[treatment_col].mean())
            z = 1.645 # Simple z for 95%
            ss_a = z * d_a.std() * np.sqrt(sim_lt)
            ts_a = np.full(len(d_a), (d_a.mean() * sim_lt) + ss_a)
            
            # Universe B logic
            sim_df_b = test_df.copy(); sim_df_b[f'New_{treatment_col}'] = price_b
            d_b = engine.predict_counterfactual(sim_df_b, f'New_{treatment_col}')
            ss_b = z * (test_df[target_col] - engine.base_model.predict(test_df[engine.features])).std() * np.sqrt(sim_lt)
            ts_b = (d_b * sim_lt) + ss_b
            
            inv_a, short_a, lost_a = simulate_inventory_dynamic(d_a, ts_a, sim_lt)
            inv_b, short_b, lost_b = simulate_inventory_dynamic(d_b, ts_b, sim_lt)
            
            rev_a = ((d_a - lost_a) * p_a).sum(); rev_b = ((d_b - lost_b) * price_b).sum()

            def display_bilingual_comp(label, v_a, v_b, fmt, higher_better=True):
                delta = v_b - v_a
                color = "comp-delta-pos" if (delta >= 0 if higher_better else delta <= 0) else "comp-delta-neg"
                st.markdown(f"""<div style="background:white; padding:10px; border:1px solid #ddd; border-radius:10px;">
                    <div class='comp-label'>{label}</div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Univ A: {fmt(v_a)}</span><span style="color:#0ea5e9">Univ B: {fmt(v_b)}</span>
                    </div>
                    <div class='{color}' style='text-align:center;'>{LANG_DICT['improvement']}: {delta:+.1f}</div>
                </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1: display_bilingual_comp(LANG_DICT["rev_comp"], rev_a, rev_b, lambda x: f"${x:,.0f}")
            with c2: display_bilingual_comp(LANG_DICT["inv_comp"], np.mean(inv_a), np.mean(inv_b), lambda x: f"{x:.0f}", False)
            with c3: display_bilingual_comp(LANG_DICT["sl_comp"], 100*(1-short_a/len(test_df)), 100*(1-short_b/len(test_df)), lambda x: f"{x:.1f}%")

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(y=inv_a, name=LANG_DICT["u_a"], line=dict(color='#94a3b8')))
            fig_p.add_trace(go.Scatter(y=inv_b, name=LANG_DICT["u_b"], line=dict(color='#0ea5e9')))
            st.plotly_chart(fig_p, use_container_width=True)

    with tab6:
        st.subheader(LANG_DICT["tab6"])
        st.write("Ready to generate report / 報告已準備就緒。")
        if st.button("📄 Download PDF / 下載報告"):
            st.success("PDF Generated (Simulated) / PDF 已生成。")