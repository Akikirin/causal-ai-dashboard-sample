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
    st.error("🚨 Please install xgboost: `pip install xgboost` / 請安裝 xgboost")
    st.stop()

try:
    from econml.dml import LinearDML
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install econml` / 缺失必要套件：請執行 pip install econml")
    st.stop()

try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install causalml` / 缺失必要套件：請執行 pip install causalml")
    st.stop()

from fpdf import FPDF

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 2. Page Configuration & Professional CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Universal Causal Dashboard", page_icon="🔮")

# Translation Helper Function
def t(en, tw):
    return f"{en} | {tw}"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans&display=swap');
    html, body, [class*="css"], font, div, span, p, text {
        font-family: 'Josefin Sans', sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Josefin Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    [data-testid="stMetricValue"] { color: #000000 !important; font-family: 'Josefin Sans', sans-serif !important; }
    .stTabs [data-baseweb="tab"] { font-family: 'Josefin Sans', sans-serif !important; font-size: 1.1rem; font-weight: 600; }
    .theory-box {
        background-color: #f0f9ff;
        border-left: 5px solid #0ea5e9;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #334155;
    }
    .comp-delta-pos { color: #059669; font-weight: bold; }
    .comp-delta-neg { color: #dc2626; font-weight: bold; }
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
            random_state=42,
            cv=3
        )
        self.base_model = XGBRegressor(n_estimators=100, random_state=42)
        self.features = []
        self.treatment = ""
        self.confounders = []

    def train(self, df, target_col, treatment_col, confounders, heterogeneity_cols=None):
        self.target = target_col
        self.treatment = treatment_col
        self.confounders = confounders
        X = df[heterogeneity_cols] if heterogeneity_cols else None
        W = df[confounders]
        Y = df[target_col]
        T = df[treatment_col]

        with st.spinner(t("🧠 Engines warming up... DML running 3-Fold Cross-Fitting...", "🧠 引擎啟動中... 正在進行 DML 三折交叉驗證...")):
            self.dml_est.fit(Y, T, X=X, W=W)
            all_feats = [treatment_col] + confounders + (heterogeneity_cols if heterogeneity_cols else [])
            self.base_model.fit(df[all_feats], Y)
            self.features = all_feats

    def get_causal_effect(self, X_pred):
        return self.dml_est.effect(X_pred)

    def predict_counterfactual(self, df_input, new_price_col):
        base_pred = self.base_model.predict(df_input[self.features])
        delta_t = df_input[new_price_col] - df_input[self.treatment]
        if 'Latent_Market_State' in df_input.columns:
            theta = self.dml_est.effect(df_input[['Latent_Market_State']])
        else:
            theta = self.dml_est.const_marginal_effect(df_input[self.confounders])
        counterfactual_sales = base_pred + (theta * delta_t)
        return np.maximum(counterfactual_sales, 0)

def train_meta_learners(df, target_col, treatment_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    w = df[treatment_col].copy()
    if w.nunique() > 2:
        median_val = w.median()
        w_binary = (w > median_val).astype(int)
    else:
        w_binary = w.astype(int)

    results = {}
    learner_s = BaseSRegressor(learner=LinearRegression())
    cate_s = learner_s.fit_predict(X=X, treatment=w_binary, y=y)
    results['S-Learner'] = cate_s.flatten()

    learner_t = BaseTRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0))
    cate_t = learner_t.fit_predict(X=X, treatment=w_binary, y=y)
    results['T-Learner'] = cate_t.flatten()
    
    learner_x = BaseXRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0))
    cate_x = learner_x.fit_predict(X=X, treatment=w_binary, y=y)
    results['X-Learner'] = cate_x.flatten()
    return pd.DataFrame(results)

# ==========================================
# 5. Main Application Logic
# ==========================================
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title(t("🧠 Causal AI Strategy Dashboard", "🧠 因果 AI 策略儀表板"))
    st.markdown(t("Quantify the **True Impact** of your decisions using Double Machine Learning.", "利用 **雙重機器學習 (DML)** 量化決策的 **真實影響力**。"))

with st.sidebar:
    st.header(t("🎛️ Control Tower", "🎛️ 控制台"))
    st.info(t("Upload your historical sales data to begin causal inference.", "上傳歷史銷售數據以開始因果推斷。"))
    uploaded_file = st.file_uploader(t("Upload CSV Data", "上傳 CSV 數據"), type="csv")
    
    if uploaded_file:
        raw_df = load_data(uploaded_file)
        cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        st.markdown(t("### 1. Model Configuration", "### 1. 模型配置"))
        target_col = st.selectbox(t("🎯 Target (Outcome Y)", "🎯 目標變數 (結果 Y)"), cols, index=0)
        treatment_col = st.selectbox(t("💊 Treatment (Input T)", "💊 干預變數 (輸入 T)"), cols, index=1)
        avail_cols = [c for c in cols if c not in [target_col, treatment_col]]
        confounders = st.multiselect(t("🌪️ Confounders (Controls W)", "🌪️ 混雜因子 (控制變數 W)"), avail_cols, default=avail_cols[:2])
        
        st.markdown(t("### 2. Execution", "### 2. 執行"))
        if st.button(t("🚀 Run Causal Engine", "🚀 啟動因果引擎"), type="primary", use_container_width=True):
            st.session_state['run'] = True
            st.session_state['cate_results'] = None
            st.session_state['fold_metrics'] = None
            st.session_state['ols_fold_metrics'] = None
            st.session_state['sim_results'] = None
    else:
        st.caption(t("Waiting for data...", "等待數據上傳..."))

# --- Main Content ---
if st.session_state.get('run', False) and uploaded_file:
    df_eng = auto_feature_eng(raw_df, target_col, treatment_col)
    train_size = int(len(df_eng) * 0.8)
    train_df = df_eng.iloc[:train_size]
    test_df = df_eng.iloc[train_size:].reset_index(drop=True)
    all_confounders = confounders + [c for c in df_eng.columns if 'Lag' in c]

    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(t("Observation Window", "觀察窗口"), f"{len(df_eng)} " + t("Periods", "週期"))
    m_col2.metric(t("Target Variable", "目標變數"), target_col)
    m_col3.metric(t("Treatment Variable", "干預變數"), treatment_col)
    m_col4.metric(t("Confounders Tracked", "追蹤混雜因子數"), len(all_confounders))
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t("⚡ Insights & Elasticity", "⚡ 洞察與彈性分析"),
        t("🔮 Sensitivity Simulator", "🔮 敏感度模擬器"),
        t("⚔️ Model Battle", "⚔️ 模型競技場"),
        t("⚖️ Evaluation", "⚖️ 模型評估"),
        t("🌌 Parallel Universe", "🌌 平行時空模擬"),
        t("📋 Executive Report", "📋 執行報告")
    ])
    
    with tab1:
        st.subheader(t("Separating Signal from Noise", "撥開雲霧見青天：區分信號與雜訊"))
        c_dag, c_explain = st.columns([1, 2])
        with c_dag:
            dot = graphviz.Digraph()
            dot.attr(rankdir='LR', size='8,5')
            dot.attr('node', shape='box', style='filled,rounded', fontname='Inter')
            dot.node('T', t('Treatment', '干預決策'), fillcolor='#d1fae5', color='#059669')
            dot.node('Y', t('Outcome', '最終結果'), fillcolor='#dbeafe', color='#2563eb')
            dot.node('W', t('Confounders', '混雜因子'), shape='ellipse', fillcolor='#fee2e2', color='#dc2626')
            dot.edge('T', 'Y', label=t(' Causal Link', ' 因果鏈結'), color='#059669', penwidth='2.0')
            dot.edge('W', 'T', style='dashed', color='#94a3b8')
            dot.edge('W', 'Y', style='dashed', color='#94a3b8')
            st.graphviz_chart(dot)
        with c_explain:
            st.markdown(f"""<div class='theory-box'><b>{t("The 'Noise Cancellation' Logic:", "「雜訊消除」邏輯：")}</b><br>
            {t("Standard correlations are biased because <b>Confounders</b> (Red) affect both your decision (T) and the outcome (Y).", "標準相關性分析存在偏誤，因為 <b>混雜因子</b> (紅色) 同時影響你的決策 (T) 與結果 (Y)。")}<br><br>
            {t("We use <b>Double Machine Learning</b> to 'block' the red dashed lines, isolating the pure green causal link.", "我們使用 <b>雙重機器學習 (DML)</b> 來「阻斷」紅色虛線，從而分離出純粹的綠色因果鏈結。")}</div>""", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric(t("True Causal Elasticity", "真實因果彈性"), f"{avg_elasticity:.3f}", help=t("The actual impact of Treatment on Target, free of bias.", "干預對目標的實際影響，已排除偏誤。"))
        k2.metric(t("Naive Correlation", "原始相關性"), f"{naive_corr:.3f}", delta=f"{t('Bias Detected', '偵測到偏誤')}: {bias_delta:.3f}", delta_color="inverse")
        bias_status = t("Significant Bias", "顯著偏誤") if abs(bias_delta) > 0.1 else t("Clean Data", "數據純淨")
        k3.metric(t("Data Reliability", "數據可靠性"), bias_status, delta=t("Corrected via DML", "已透過 DML 修正"))

        viz_df = pd.DataFrame({'Market Momentum': test_df['Latent_Market_State'], 'Impact': effects})
        fig_hte = px.scatter(viz_df, x='Market Momentum', y='Impact', color='Impact', color_continuous_scale='Tealgrn', title=t("Heterogeneous Treatment Effects", "異質性干預效果分析"))
        fig_hte.update_layout(template="plotly_white", xaxis_title=t("Market Momentum (PCA)", "市場動能 (PCA)"), yaxis_title=t("Causal Impact", "因果影響"))
        st.plotly_chart(fig_hte, use_container_width=True)

    with tab2:
        st.subheader(t("🔮 Multi-Scenario Simulator", "🔮 多情境模擬器"))
        col_in, col_out = st.columns([1, 2])
        with col_in:
            st.markdown(t("### 🛠️ Adjust Strategy", "### 🛠️ 調整策略"))
            curr_avg = float(test_df[treatment_col].mean())
            price_main = st.slider(t("Proposed Treatment Value (Center)", "建議干預值 (中心)"), min_value=float(test_df[treatment_col].min()), max_value=float(test_df[treatment_col].max()), value=curr_avg)
            comp_mode = st.radio(t("Comparison Mode", "比較模式"), [t("Percentage (+/- %)", "百分比 (+/- %)"), t("Manual Prices ($)", "手動輸入數值 ($)")], horizontal=True)
            if "Percentage" in comp_mode or "百分比" in comp_mode:
                sensitivity = st.slider(t("Comparison Interval (+/- %)", "比較區間 (+/- %)"), 1, 20, 5)
                price_low, price_high = price_main * (1 - sensitivity/100), price_main * (1 + sensitivity/100)
                scenario_labels = [t(f"Lower (-{sensitivity}%)", f"較低 (-{sensitivity}%)"), t("Proposed", "建議方案"), t(f"Higher (+{sensitivity}%)", f"較高 (+{sensitivity}%)")]
            else:
                c1, c2 = st.columns(2)
                price_low = c1.number_input(t("Lower Scenario", "較低情境"), value=float(price_main*0.95))
                price_high = c2.number_input(t("Higher Scenario", "較高情境"), value=float(price_main*1.05))
                scenario_labels = [t("Scenario A (Low)", "情境 A (低)"), t("Proposed", "建議方案"), t("Scenario B (High)", "情境 B (高)")]
            st.markdown(t("### 📦 Inventory Specs", "### 📦 庫存規格"))
            lead_time = st.number_input(t("Lead Time (Days)", "前置時間 (天)"), value=5)

        with col_out:
            sim_df_main = test_df.copy(); sim_df_main[f'New_{treatment_col}'] = price_main
            cf_main = engine.predict_counterfactual(sim_df_main, f'New_{treatment_col}')
            
            total_act = test_df[target_col].sum()
            total_sim = cf_main.sum()
            rev_sim = total_sim * price_main

            s1, s2, s3 = st.columns(3)
            s1.metric(t("Projected Demand", "預測需求量"), f"{total_sim:,.0f}", delta=f"{(total_sim-total_act):,.0f}")
            s2.metric(t("Projected Value", "預測營收"), f"${rev_sim:,.0f}", delta=f"${(rev_sim - (total_act*curr_avg)):,.0f}")
            st.info(t("Detailed scenario breakdown shown in report tab.", "詳細情境分析顯示於報告分頁中。"))

            fig_cf = go.Figure()
            fig_cf.add_trace(go.Scatter(y=test_df[target_col], name=t("Historical Actuals", "歷史實際值"), line=dict(color='#cbd5e1')))
            fig_cf.add_trace(go.Scatter(y=cf_main, name=t("Causal Prediction", "因果預測"), line=dict(color='#0ea5e9', width=4)))
            fig_cf.update_layout(title=t("Strategic Impact Visualization", "策略影響視覺化"), template="plotly_white")
            st.plotly_chart(fig_cf, use_container_width=True)

    with tab3:
        st.subheader(t("⚔️ Battle of the Meta-Learners", "⚔️ 元學習器競技場"))
        if st.session_state.get('cate_results') is None:
             if st.button(t("🏁 Start Tournament", "🏁 開始競賽"), use_container_width=True):
                 meta_feats = all_confounders + ['Latent_Market_State']
                 with st.spinner(t("Running Causal Tournament...", "因果模型競賽進行中...")):
                    st.session_state['cate_results'] = train_meta_learners(df_eng, target_col, treatment_col, meta_feats)
        
        if st.session_state.get('cate_results') is not None:
            cate_results = st.session_state['cate_results']
            fig_hist = px.histogram(cate_results, barmode='overlay', title=t("Distribution of Causal Estimates", "因果估計值分布"))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.success(t("Analysis complete. X-Learner is typically the most robust for unbalanced data.", "分析完成。對於不平衡數據，X-Learner 通常最為穩健。"))

    with tab4:
        st.subheader(t("⚖️ Methodology Evaluation", "⚖️ 算法評估"))
        st.markdown(f"""<div class='theory-box'><b>{t("Why DML? (3-Fold Cross-Fitting)", "為什麼選擇 DML？(三折交叉驗證)")}</b><br>
        {t("Standard models confuse correlation with causation. To fix this, we use the <b>Frisch-Waugh-Lovell (FWL)</b> theorem.", "標準模型常混淆相關性與因果關係。為了修正這點，我們採用 <b>FWL 定理</b>。")}</div>""", unsafe_allow_html=True)
        if st.session_state.get('fold_metrics') is None:
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            fold_metrics, ols_fold_metrics = [], []
            with st.spinner(t("Running 3-Fold Stability Check...", "正在執行三折穩定性檢查...")):
                for train_idx, val_idx in kf.split(train_df):
                    X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
                    f_engine = RealCausalEngine()
                    f_engine.train(X_tr, target_col, treatment_col, all_confounders, heterogeneity_cols=['Latent_Market_State'])
                    fold_metrics.append(np.mean(f_engine.get_causal_effect(X_val[['Latent_Market_State']])))
                    ols = LinearRegression().fit(X_tr[[treatment_col] + all_confounders], X_tr[target_col])
                    ols_fold_metrics.append(ols.coef_[0])
            st.session_state['fold_metrics'] = fold_metrics
            st.session_state['ols_fold_metrics'] = ols_fold_metrics
        
        fig_val = go.Figure()
        fig_val.add_trace(go.Bar(x=['Fold 1', 'Fold 2', 'Fold 3'], y=st.session_state['fold_metrics'], name='DML (Causal)', marker_color='#0ea5e9'))
        fig_val.add_trace(go.Bar(x=['Fold 1', 'Fold 2', 'Fold 3'], y=st.session_state['ols_fold_metrics'], name='OLS (Traditional)', marker_color='#ef4444'))
        st.plotly_chart(fig_val, use_container_width=True)

    with tab5:
        st.subheader(t("🌌 Parallel Universe Simulation", "🌌 平行時空模擬"))
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            st.markdown(t("### ⚙️ Universe B Settings", "### ⚙️ 平行時空 B 配置"))
            price_b = st.slider(t("Universe B Price ($)", "時空 B 數值 ($)"), min_value=float(test_df[treatment_col].min()), max_value=float(test_df[treatment_col].max()), value=float(test_df[treatment_col].mean()))
            target_sl = st.slider(t("Target Service Level (%)", "目標服務水準 (%)"), 90, 99, 95) / 100
            sim_lt = st.number_input(t("Supply Lead Time (Days)", "供應前置時間 (天)"), value=5, key="plt")
        
        with col_p2:
            # Calculation logic for Universe A vs B
            z_score = norm.ppf(target_sl)
            demand_a = test_df[target_col]
            ss_a = z_score * demand_a.std() * np.sqrt(sim_lt)
            ts_a = np.full(len(demand_a), (demand_a.mean() * sim_lt) + ss_a)
            
            sim_df_b = test_df.copy(); sim_df_b[f'New_{treatment_col}'] = price_b
            demand_b = engine.predict_counterfactual(sim_df_b, f'New_{treatment_col}')
            preds_b = engine.base_model.predict(test_df[engine.features])
            ss_b = z_score * (test_df[target_col] - preds_b).std() * np.sqrt(sim_lt)
            ts_b = (demand_b * sim_lt) + ss_b

            inv_a, short_a, lost_a = simulate_inventory_dynamic(demand_a, ts_a, sim_lt)
            inv_b, short_b, lost_b = simulate_inventory_dynamic(demand_b, ts_b, sim_lt)
            
            rev_a = ((demand_a - lost_a) * float(test_df[treatment_col].mean())).sum()
            rev_b = ((demand_b - lost_b) * price_b).sum()

            st.session_state['sim_results'] = {'rev_a': rev_a, 'rev_b': rev_b, 'inv_a': inv_a.mean(), 'inv_b': inv_b.mean(), 'short_a': short_a, 'short_b': short_b}

            c1, c2, c3 = st.columns(3)
            c1.metric(t("Revenue B vs A", "營收：時空 B vs A"), f"${rev_b:,.0f}", delta=f"${(rev_b-rev_a):,.0f}")
            c2.metric(t("Avg Inventory B", "時空 B 平均庫存"), f"{inv_b.mean():,.0f}", delta=f"{(inv_b.mean()-inv_a.mean()):,.0f}", delta_color="inverse")
            c3.metric(t("Shortage Events B", "時空 B 缺貨次數"), f"{short_b} days", delta=f"{short_b-short_a}", delta_color="inverse")

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(y=inv_a, name=t("Universe A (Static)", "時空 A (靜態)"), line=dict(color='#94a3b8')))
            fig_p.add_trace(go.Scatter(y=inv_b, name=t("Universe B (Dynamic)", "時空 B (動態)"), line=dict(color='#0ea5e9')))
            st.plotly_chart(fig_p, use_container_width=True)

    with tab6:
        st.subheader(t("📋 Executive Summary", "📋 決策執行摘要"))
        sim = st.session_state.get('sim_results', {})
        
        report_txt = f"""
        {t("CAUSAL AI EXECUTIVE SUMMARY", "因果 AI 執行摘要")}
        ===========================
        
        1. {t("EXECUTIVE FINDINGS", "核心分析發現")}
        ---------------------
        - {t("True Causal Elasticity", "真實因果彈性")}: {avg_elasticity:.4f}
        - {t("Naive Correlation", "原始相關性")}: {naive_corr:.4f}
        - {t("Bias Detected", "偵測偏誤")}: {bias_delta:.4f}
        
        2. {t("PARALLEL UNIVERSE RESULTS", "平行時空模擬結果")}
        -----------------------------------------
        - {t("Revenue", "營營收")}: ${sim.get('rev_a',0):,.0f} (A) vs ${sim.get('rev_b',0):,.0f} (B)
        - {t("Shortages", "缺貨天數")}: {sim.get('short_a',0)} (A) vs {sim.get('short_b',0)} (B)
        
        3. {t("STRATEGIC RECOMMENDATION", "戰略建議")}
        ---------------------------
        {t("Focus on 'High Momentum' periods to adjust strategy.", "專注於「高動能」時期來調整策略。")}
        """
        st.text_area(t("Report Preview", "報告預覽"), report_txt, height=300)
        if st.button(t("📄 Download PDF Report", "📄 下載 PDF 報告")):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, report_txt.encode('latin-1', 'replace').decode('latin-1'))
            st.download_button("Download PDF", pdf.output(dest='S'), "report.pdf")