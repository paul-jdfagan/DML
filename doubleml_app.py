import streamlit as st
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from flaml import AutoML
import doubleml as dml
import matplotlib.pyplot as plt
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="DoubleML Causal Analysis", layout="wide", page_icon="📊")

# Custom RMSE function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Generic temporal transformer
class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Generic temporal feature engineering for time series data"""
    
    def __init__(self, date_col="Date", include_cyclical=True, include_calendar=True, 
                 include_trend=True, include_retail_events=False):
        self.date_col = date_col
        self.include_cyclical = include_cyclical
        self.include_calendar = include_calendar
        self.include_trend = include_trend
        self.include_retail_events = include_retail_events

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        
        features_added = []

        # Cyclical encodings
        if self.include_cyclical:
            df["month_sin"] = np.sin(2*np.pi * df[self.date_col].dt.month/12)
            df["month_cos"] = np.cos(2*np.pi * df[self.date_col].dt.month/12)
            df["dow_sin"] = np.sin(2*np.pi * df[self.date_col].dt.weekday/7)
            df["dow_cos"] = np.cos(2*np.pi * df[self.date_col].dt.weekday/7)
            df["doy_sin"] = np.sin(2*np.pi * df[self.date_col].dt.dayofyear/365)
            df["doy_cos"] = np.cos(2*np.pi * df[self.date_col].dt.dayofyear/365)
            features_added.extend(["month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"])

        # Calendar flags
        if self.include_calendar:
            df["is_weekend"] = (df[self.date_col].dt.weekday >= 5).astype(int)
            df["is_month_start"] = (df[self.date_col].dt.day <= 7).astype(int)
            df["is_month_end"] = (df[self.date_col].dt.day >= 24).astype(int)
            df["is_quarter_start"] = df[self.date_col].dt.month.isin([1,4,7,10]).astype(int)
            df["is_quarter_end"] = df[self.date_col].dt.month.isin([3,6,9,12]).astype(int)
            features_added.extend(["is_weekend", "is_month_start", "is_month_end", 
                                 "is_quarter_start", "is_quarter_end"])

        # Retail/shopping season flags (optional)
        if self.include_retail_events:
            df["is_holiday_season"] = (
                (df[self.date_col].dt.month==12) |
                ((df[self.date_col].dt.month==11)&(df[self.date_col].dt.day>=15))
            ).astype(int)
            df["is_black_friday_week"] = (
                (df[self.date_col].dt.month==11)&(df[self.date_col].dt.day>=20)
            ).astype(int)
            df["is_nordstrom_anniversary_sale"] = (
                ((df[self.date_col].dt.month==7)&(df[self.date_col].dt.day>=15)) |
                ((df[self.date_col].dt.month==8)&(df[self.date_col].dt.day<=10))
            ).astype(int)
            df["is_sephora_spring_sale"] = (
                (df[self.date_col].dt.month==4)&
                df[self.date_col].dt.day.between(1,20)
            ).astype(int)
            df["is_walmart_july_event"] = (
                (df[self.date_col].dt.month==7)&
                df[self.date_col].dt.day.between(10,20)
            ).astype(int)
            df["is_target_july_event"] = df["is_walmart_july_event"]
            df["is_amazon_prime_days_window"] = (
                (df[self.date_col].dt.month==7)&
                df[self.date_col].dt.day.between(9,18)
            ).astype(int)
            features_added.extend([
                "is_holiday_season", "is_black_friday_week", "is_nordstrom_anniversary_sale",
                "is_sephora_spring_sale", "is_walmart_july_event", "is_target_july_event",
                "is_amazon_prime_days_window"
            ])

        # Trend features
        if self.include_trend:
            df["time_trend"] = np.arange(len(df))
            df["time_trend_sq"] = df["time_trend"] ** 2
            features_added.extend(["time_trend", "time_trend_sq"])

        return df.drop(columns=[self.date_col]), features_added

def geometric_adstock(s, alpha=0.9, lags=7):
    """Apply geometric adstock transformation"""
    w = alpha ** np.arange(lags)
    return s.rolling(lags, min_periods=1).apply(lambda x: np.dot(x[::-1], w[:len(x)]), raw=True)

def apply_lag_features(df, col, lags=[1, 7, 14, 30]):
    """Apply simple lag features"""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'dml_plr' not in st.session_state:
    st.session_state.dml_plr = None

# Title
st.title("DoubleML Causal Analysis Tool")
st.markdown("*Flexible time series causal inference for any domain*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    step = st.radio(
        "Select Step:",
        ["1️⃣ Upload Data", "2️⃣ Configure Analysis", "3️⃣ Run Analysis", "4️⃣ Sensitivity Analysis"],
        key="nav_step"
    )
    st.markdown("---")
    st.info("💡 **Tip**: Follow the steps in order for best results.")
    
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool implements **Double Machine Learning (DoubleML)** for causal inference on time series data.
        
        **Use cases:**
        - Marketing: Ad spend → Sales
        - Healthcare: Treatment → Outcomes
        - Policy: Intervention → Impact
        - Finance: Events → Returns
        - Operations: Changes → Metrics
        """)

# Step 1: Upload Data
if step == "1️⃣ Upload Data":
    st.header("Step 1: Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    with col2:
        st.info("**Requirements:**\n- Time series data\n- Date column\n- Outcome variable\n- Treatment variable")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Dataset uploaded successfully! {len(df)} rows × {len(df.columns)} columns")
        
        tab1, tab2, tab3 = st.tabs(["📊 Preview", "📈 Summary Stats", "🔍 Data Quality"])
        
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numeric Columns:**")
                st.dataframe(df.describe())
            with col2:
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']))
        
        with tab3:
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.warning(f"⚠️ Missing values detected in {(missing > 0).sum()} columns:")
                missing_df = pd.DataFrame({
                    'Column': missing[missing > 0].index,
                    'Missing Count': missing[missing > 0].values,
                    'Missing %': (missing[missing > 0].values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values detected")
            
            # Check for potential date columns
            potential_dates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_dates:
                st.info(f"🗓️ Potential date columns detected: {', '.join(potential_dates)}")
        
    else:
        st.info("👆 Please upload a CSV file to begin")

# Step 2: Configure Analysis
elif step == "2️⃣ Configure Analysis":
    st.header("Step 2: Configure Your Analysis")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first (Step 1)")
    else:
        df = st.session_state.df
        
        # Core Variables Section
        st.subheader("🎯 Core Variables")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox("📅 Date Column", df.columns.tolist(), key="date_col")
        
        with col2:
            outcome_col = st.selectbox("📊 Outcome Variable (Y)", 
                                      [col for col in df.columns if col != date_col], 
                                      key="outcome_col")
        
        with col3:
            treatment_col = st.selectbox("💉 Treatment Variable (D)", 
                                        [col for col in df.columns if col not in [date_col, outcome_col]], 
                                        key="treatment_col")
        
        # Confounders Section
        st.markdown("---")
        st.subheader("🔧 Confounding Variables")
        
        available_cols = [col for col in df.columns if col not in [date_col, outcome_col, treatment_col]]
        confounder_cols = st.multiselect(
            "Select confounding variables (X) - variables that affect both treatment and outcome",
            available_cols,
            help="These are the observed covariates that may confound the causal relationship",
            key="confounder_cols"
        )
        
        # Treatment Transformation Section
        st.markdown("---")
        st.subheader("🔄 Treatment Transformation (Optional)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            treatment_transform = st.selectbox(
                "Transform treatment variable?",
                ["None", "Adstock (Geometric Decay)", "Simple Lags"],
                help="Apply transformations to capture delayed/cumulative effects"
            )
        
        with col2:
            if treatment_transform == "Adstock (Geometric Decay)":
                st.info("**Adstock** models carryover effects (common in marketing/advertising)")
                col2a, col2b = st.columns(2)
                with col2a:
                    alpha = st.slider("Decay Rate (α)", 0.0, 1.0, 0.9, 0.05, 
                                     help="How quickly effects decay (higher = slower decay)")
                with col2b:
                    lags = st.slider("Number of Lags", 1, 30, 7,
                                    help="How many periods to look back")
            elif treatment_transform == "Simple Lags":
                st.info("**Simple Lags** use past values as features")
                lag_periods = st.multiselect("Lag periods", [1, 7, 14, 30, 60, 90], default=[1, 7])
        
        # Feature Engineering Section
        st.markdown("---")
        st.subheader("🛠️ Feature Engineering")
        
        st.write("Select which temporal features to automatically generate:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            include_cyclical = st.checkbox("📈 Cyclical Features", value=True,
                                          help="Sin/cos encodings for month, day of week, day of year")
        
        with col2:
            include_calendar = st.checkbox("📅 Calendar Features", value=True,
                                          help="Weekend, month start/end, quarter flags")
        
        with col3:
            include_trend = st.checkbox("📉 Trend Features", value=True,
                                       help="Linear and quadratic time trends")
        
        with col4:
            include_retail = st.checkbox("🛍️ Retail Events", value=False,
                                        help="Holiday season, Black Friday, Prime Day, etc.")
        
        # Model Configuration Section
        st.markdown("---")
        st.subheader("⚙️ Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_folds = st.slider("CV Folds", 2, 10, 5, 
                               help="Number of time series cross-validation folds")
        
        with col2:
            time_budget = st.slider("FLAML Time Budget (s)", 30, 600, 120, 30,
                                   help="Time budget per model (outcome & treatment)")
        
        with col3:
            estimators = st.multiselect(
                "ML Estimators",
                ["lgbm", "xgboost", "histgb", "xgb_limitdepth", "catboost", "rf", "extra_tree"],
                default=["lgbm", "xgboost", "histgb"],
                help="Machine learning algorithms to try"
            )
        
        # Advanced Options
        with st.expander("🔬 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                score_type = st.selectbox(
                    "DML Score Type",
                    ["partialling out", "IV-type"],
                    help="Orthogonalization approach for DML"
                )
                
                n_rep = st.slider("Number of Repetitions", 1, 10, 1,
                                 help="Repeated sample splitting for stability")
            
            with col2:
                dml_procedure = st.selectbox(
                    "DML Procedure",
                    ["dml1", "dml2"],
                    index=1,
                    help="DML1: no cross-fitting, DML2: with cross-fitting"
                )
                
                trim_threshold = st.slider("Propensity Score Trimming", 0.0, 0.2, 0.01, 0.01,
                                          help="Trim extreme propensity scores (0 = no trimming)")
        
        # Save Configuration
        st.markdown("---")
        if st.button("✅ Confirm Configuration", type="primary", use_container_width=True):
            
            # Validate configuration
            if not confounder_cols:
                st.warning("⚠️ Consider adding confounding variables for more robust estimates")
            
            config = {
                'date_col': date_col,
                'outcome_col': outcome_col,
                'treatment_col': treatment_col,
                'confounder_cols': confounder_cols,
                'treatment_transform': treatment_transform,
                'n_folds': n_folds,
                'time_budget': time_budget,
                'estimators': estimators,
                'include_cyclical': include_cyclical,
                'include_calendar': include_calendar,
                'include_trend': include_trend,
                'include_retail': include_retail,
                'score_type': score_type,
                'n_rep': n_rep,
                'dml_procedure': dml_procedure,
                'trim_threshold': trim_threshold
            }
            
            # Add transformation-specific params
            if treatment_transform == "Adstock (Geometric Decay)":
                config['alpha'] = alpha
                config['lags'] = lags
            elif treatment_transform == "Simple Lags":
                config['lag_periods'] = lag_periods
            
            st.session_state.variable_config = config
            
            # Show configuration summary
            st.success("✅ Configuration saved! Here's your setup:")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Variables:**")
                st.write(f"- Outcome: `{outcome_col}`")
                st.write(f"- Treatment: `{treatment_col}`")
                st.write(f"- Confounders: {len(confounder_cols)}")
            
            with summary_col2:
                st.write("**Configuration:**")
                st.write(f"- Transform: {treatment_transform}")
                st.write(f"- CV Folds: {n_folds}")
                st.write(f"- Time Budget: {time_budget}s")
            
            st.info("👉 Proceed to Step 3 to run the analysis")

# Step 3: Run Analysis
elif step == "3️⃣ Run Analysis":
    st.header("Step 3: Run Causal Analysis")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first (Step 1)")
    elif 'variable_config' not in st.session_state:
        st.warning("⚠️ Please configure your analysis first (Step 2)")
    else:
        config = st.session_state.variable_config
        
        # Configuration Summary
        st.subheader("📋 Configuration Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Outcome", config['outcome_col'])
            st.metric("Treatment", config['treatment_col'])
        
        with col2:
            st.metric("Confounders", len(config['confounder_cols']))
            st.metric("Transform", config['treatment_transform'].split()[0])
        
        with col3:
            st.metric("CV Folds", config['n_folds'])
            st.metric("Estimators", len(config['estimators']))
        
        with col4:
            st.metric("Time Budget", f"{config['time_budget']}s")
            st.metric("DML Score", config['score_type'].split()[0])
        
        st.markdown("---")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis... This may take several minutes."):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Load and preprocess
                    status_text.text("Step 1/7: Loading and preprocessing data...")
                    progress_bar.progress(5)
                    
                    df = st.session_state.df.copy()
                    
                    # Handle missing values in treatment
                    if df[config['treatment_col']].isnull().any():
                        df[config['treatment_col']].fillna(method='ffill', inplace=True)
                    
                    # Step 2: Apply treatment transformation
                    status_text.text("Step 2/7: Applying treatment transformation...")
                    progress_bar.progress(15)
                    
                    treatment_var_name = config['treatment_col']
                    
                    if config['treatment_transform'] == "Adstock (Geometric Decay)":
                        df["transformed_treatment"] = geometric_adstock(
                            df[config['treatment_col']], 
                            alpha=config['alpha'], 
                            lags=config['lags']
                        )
                        treatment_var_name = "transformed_treatment"
                    elif config['treatment_transform'] == "Simple Lags":
                        df = apply_lag_features(df, config['treatment_col'], config['lag_periods'])
                        # Keep original as primary, but lags will be in confounders
                        treatment_var_name = config['treatment_col']
                    else:
                        df["transformed_treatment"] = df[config['treatment_col']]
                        treatment_var_name = "transformed_treatment"
                    
                    # Drop rows with missing outcome or treatment
                    df.dropna(subset=[config['outcome_col'], treatment_var_name], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    
                    # Step 3: Feature engineering
                    status_text.text("Step 3/7: Engineering temporal features...")
                    progress_bar.progress(25)
                    
                    transformer = TemporalFeatures(
                        date_col=config['date_col'],
                        include_cyclical=config['include_cyclical'],
                        include_calendar=config['include_calendar'],
                        include_trend=config['include_trend'],
                        include_retail_events=config['include_retail']
                    )
                    
                    df_temp, temp_feats = transformer.fit_transform(df)
                    
                    # Combine all features
                    all_x = config['confounder_cols'] + temp_feats
                    
                    # Build model dataframe
                    df_model = pd.concat([
                        df_temp[temp_feats],
                        df[config['confounder_cols']],
                        df[[config['outcome_col'], treatment_var_name]]
                    ], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
                    
                    # Drop any remaining NaNs
                    df_model.dropna(inplace=True)
                    
                    st.info(f"📊 Analysis dataset: {len(df_model)} observations, {len(all_x)} features")
                    
                    # Step 4: Set up cross-validation
                    status_text.text("Step 4/7: Setting up time series cross-validation...")
                    progress_bar.progress(35)
                    
                    my_splitter = TimeSeriesSplit(n_splits=config['n_folds'])
                    
                    X_all = df_model[all_x]
                    y_outcome = df_model[config['outcome_col']]
                    d_treatment = df_model[treatment_var_name]
                    
                    # Step 5: Fit outcome model
                    status_text.text("Step 5/7: Fitting outcome model (ml_l)...")
                    progress_bar.progress(45)
                    
                    automl_l = AutoML()
                    automl_l.fit(
                        X_train=X_all, y_train=y_outcome,
                        task="regression", metric="rmse", 
                        time_budget=config['time_budget'],
                        split_type=my_splitter, 
                        estimator_list=config['estimators'],
                        verbose=0
                    )
                    
                    # Step 6: Fit treatment model
                    status_text.text("Step 6/7: Fitting treatment model (ml_m)...")
                    progress_bar.progress(60)
                    
                    automl_m = AutoML()
                    automl_m.fit(
                        X_train=X_all, y_train=d_treatment,
                        task="regression", metric="rmse", 
                        time_budget=config['time_budget'],
                        split_type=my_splitter, 
                        estimator_list=config['estimators'],
                        verbose=0
                    )
                    
                    # Step 7: Generate CV predictions
                    status_text.text("Step 7/7: Generating cross-validated predictions and computing causal effect...")
                    progress_bar.progress(75)
                    
                    cv_preds_l = np.full_like(y_outcome, fill_value=np.nan, dtype=float)
                    cv_preds_m = np.full_like(d_treatment, fill_value=np.nan, dtype=float)
                    
                    best_estimator_l = automl_l.model.estimator
                    best_estimator_m = automl_m.model.estimator
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(my_splitter.split(X_all)):
                        best_estimator_l.fit(X_all.iloc[train_idx], y_outcome.iloc[train_idx])
                        cv_preds_l[test_idx] = best_estimator_l.predict(X_all.iloc[test_idx])
                        best_estimator_m.fit(X_all.iloc[train_idx], d_treatment.iloc[train_idx])
                        cv_preds_m[test_idx] = best_estimator_m.predict(X_all.iloc[test_idx])
                    
                    mask = ~np.isnan(cv_preds_l)
                    
                    cv_loss_l = root_mean_squared_error(y_outcome[mask], cv_preds_l[mask])
                    cv_loss_m = root_mean_squared_error(d_treatment[mask], cv_preds_m[mask])
                    
                    # Prepare DoubleML (use real X for benchmarking)
                    y_for_doubleml = y_outcome.iloc[mask]
                    d_for_doubleml = d_treatment.iloc[mask]
                    
                    # Align X with mask
                    X_for_doubleml = df_model.loc[y_for_doubleml.index, all_x]
                    
                    # Combine outcome, treatment, and confounders
                    df_for_doubleml = pd.concat(
                        [
                            pd.Series(y_for_doubleml, name=config['outcome_col']),
                            pd.Series(d_for_doubleml, name=treatment_var_name),
                            X_for_doubleml,
                        ],
                        axis=1,
                    )

                    # ✅ Create DoubleMLData with real X so sensitivity_benchmark can find them
                    dml_data = dml.DoubleMLData(
                        df_for_doubleml,
                        y_col=config['outcome_col'],
                        d_cols=treatment_var_name,
                        x_cols=all_x,
                    )
                    # Fit DoubleML
                    progress_bar.progress(90)
                    
                    ml_l_dummy = dml.utils.DMLDummyRegressor()
                    ml_m_dummy = dml.utils.DMLDummyRegressor()
                    
                    pred_dict = {
                        treatment_var_name: {
                            "ml_l": cv_preds_l[mask].reshape(-1, 1),
                            "ml_m": cv_preds_m[mask].reshape(-1, 1),
                        }
                    }
                    
                    dml_plr = dml.DoubleMLPLR(
                        dml_data,
                        ml_l=ml_l_dummy,
                        ml_m=ml_m_dummy,
                        n_folds=config['n_folds'],
                        n_rep=config['n_rep'],
                        score=config['score_type']
                    )
                    
                    dml_plr.fit(external_predictions=pred_dict)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # ✅ Store for Step 4 benchmarking
                    st.session_state.results = {
                        'dml_plr': dml_plr,
                        'cv_loss_l': cv_loss_l,
                        'cv_loss_m': cv_loss_m,
                        'n_obs': dml_data.n_obs,
                        'mask_sum': mask.sum(),
                        'total_obs': len(df_model),
                        'confounder_cols': config['confounder_cols'],
                        'x_cols_for_benchmark': all_x,  # ✅ store all X cols for Step 4
                        'treatment_var_name': treatment_var_name,
                        'best_estimator_l': automl_l.best_estimator,
                        'best_estimator_m': automl_m.best_estimator,
                        'temporal_features': temp_feats
                    }
                    st.session_state.dml_plr = dml_plr
                    
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display Results
                    st.markdown("---")
                    st.header("📊 Causal Effect Estimates")
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Causal Effect (θ)", f"{dml_plr.coef[0]:.6f}",
                                 help="Average treatment effect on the outcome")
                        st.metric("Standard Error", f"{dml_plr.se[0]:.6f}")
                    
                    with col2:
                        ci_lower = dml_plr.confint().iloc[0, 0]
                        ci_upper = dml_plr.confint().iloc[0, 1]
                        st.metric("95% CI Lower", f"{ci_lower:.6f}")
                        st.metric("95% CI Upper", f"{ci_upper:.6f}")
                    
                    with col3:
                        p_val = dml_plr.pval[0]
                        st.metric("P-value", f"{p_val:.6f}")
                        if p_val < 0.05:
                            st.success("✅ Statistically significant (p < 0.05)")
                        else:
                            st.warning("⚠️ Not statistically significant (p ≥ 0.05)")
                    
                    # Interpretation
                    st.subheader("📝 Interpretation")
                    effect = dml_plr.coef[0]
                    if effect > 0:
                        st.info(f"📈 A one-unit increase in **{config['treatment_col']}** is associated with a **{effect:.4f}** unit increase in **{config['outcome_col']}** (on average, all else equal).")
                    else:
                        st.info(f"📉 A one-unit increase in **{config['treatment_col']}** is associated with a **{abs(effect):.4f}** unit decrease in **{config['outcome_col']}** (on average, all else equal).")
                    
                    # Summary table
                    st.subheader("📋 Detailed Summary")
                    summary_df = dml_plr.summary
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Model Performance
                    st.markdown("---")
                    st.subheader("🎯 Nuisance Model Performance")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        st.metric("Outcome Model RMSE", f"{cv_loss_l:.4f}",
                                 help="Lower is better - measures prediction accuracy for outcome")
                    
                    with perf_col2:
                        st.metric("Treatment Model RMSE", f"{cv_loss_m:.4f}",
                                 help="Lower is better - measures prediction accuracy for treatment")
                    
                    with perf_col3:
                        st.metric("Best Outcome Estimator", automl_l.best_estimator)
                        st.metric("Best Treatment Estimator", automl_m.best_estimator)
                    
                    # Data Processing Details
                    with st.expander("📈 Data Processing Details"):
                        st.write(f"**Original observations:** {len(df_model)}")
                        st.write(f"**Valid CV predictions:** {mask.sum()} ({100*mask.sum()/len(df_model):.1f}%)")
                        st.write(f"**Used for DoubleML:** {dml_data.n_obs}")
                        st.write(f"**Time series folds:** {config['n_folds']}")
                        st.write(f"**Total features used:** {len(all_x)}")
                        st.write(f"**Temporal features added:** {len(temp_feats)}")
                        
                        if config['treatment_transform'] != "None":
                            st.write(f"**Treatment transformation:** {config['treatment_transform']}")
                    
                    # Feature Importance (if available)
                    with st.expander("🔍 Feature Information"):
                        st.write("**Confounding Variables:**")
                        st.write(", ".join(config['confounder_cols']) if config['confounder_cols'] else "None")
                        
                        st.write("\n**Temporal Features Added:**")
                        st.write(", ".join(temp_feats))
                    
                    st.markdown("---")
                    st.info("👉 Proceed to Step 4 for sensitivity analysis to test robustness to unobserved confounding")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
                    progress_bar.empty()
                    status_text.empty()

elif step == "4️⃣ Sensitivity Analysis":
    st.header("Step 4: Sensitivity Analysis")

    # --- Guards ---
    if st.session_state.get('results') is None or st.session_state.results.get('dml_plr') is None:
        st.warning("⚠️ Please run the causal analysis first (Step 3).")
        st.info("Sensitivity analysis tests robustness to unobserved confounding.")
    else:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import doubleml as dml

        dml_plr = st.session_state.results['dml_plr']

        # Ensure a treatment label exists for titles/labels
        treatment_from_app = st.session_state.results.get('treatment_var_name', None)
        if getattr(dml_plr, "d_cols", None) and len(dml_plr.d_cols) > 0:
            treatment_names = list(dml_plr.d_cols)
        else:
            treatment_names = [treatment_from_app or "treatment"]
            dml_plr.d_cols = treatment_names
        treatment_label = treatment_names[0]

        # ---------- Plot backends ----------
        try:
            import plotly.graph_objs as go
            _PlotlyFigure = go.Figure
        except Exception:
            _PlotlyFigure = tuple()

        from matplotlib.figure import Figure as _MplFigure
        from matplotlib.axes import Axes as _MplAxes

        # --- Minimal renderer ---
        def _render(plot_obj):
            if _PlotlyFigure and isinstance(plot_obj, _PlotlyFigure):
                st.plotly_chart(plot_obj, use_container_width=True)
                return
            if isinstance(plot_obj, (_MplFigure, _MplAxes)):
                fig = plot_obj.figure if isinstance(plot_obj, _MplAxes) else plot_obj
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig)
                plt.close(fig)
                return
            st.info("Plot object not recognized; nothing to display.")

        # --- Plot styling (tutorial look) ---
        def style_plotly(fig):
            try:
                import plotly.graph_objs as go  # noqa
                fig.update_layout(
                    template="plotly_white",
                    font=dict(size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                for tr in fig.data:
                    if getattr(tr, "type", "") in ("heatmap", "contour"):
                        tr.colorscale = "Viridis"  # or "Turbo" if you prefer
                        if getattr(tr, "colorbar", None):
                            tr.colorbar.title = tr.colorbar.title or "θ bound"
                    if getattr(tr, "mode", "") and "markers" in tr.mode:
                        tr.marker.update(size=12, line=dict(width=1.5, color="white"), symbol="x")
                        if hasattr(tr, "name") and getattr(tr, "x", None) is not None:
                            tr.text = [tr.name] * len(tr.x)
                            tr.textposition = "top center"
                            tr.textfont = dict(size=11, color="black")
                return fig
            except Exception:
                return fig

        def style_and_render(fig, title_text=None):
            # Set title if Plotly; if Matplotlib, set suptitle
            if _PlotlyFigure and isinstance(fig, _PlotlyFigure):
                if title_text:
                    fig.update_layout(title=dict(text=title_text, x=0.5, xanchor="center"))
                _render(style_plotly(fig))
            elif isinstance(fig, (_MplFigure, _MplAxes)):
                _fig = fig.figure if isinstance(fig, _MplAxes) else fig
                if title_text:
                    try:
                        _fig.suptitle(title_text)
                    except Exception:
                        pass
                _render(_fig)
            else:
                _render(fig)

        st.info("📊 Sensitivity Analysis evaluates how strong unobserved confounding must be to change your conclusion "
                "(e.g., nullify θ or cross a null bound).")

        # ---------- Session-state defaults ----------
        for key, default in [
            ('last_cf_y', 0.05), ('last_cf_d', 0.05), ('last_rho', 1.0),
            ('last_level', 0.95), ('last_null_hyp', 0.0),
        ]:
            st.session_state.setdefault(key, default)

        # ============= 1) Preset Scenarios (comes first) =============
        st.subheader("📋 Preset Scenarios")
        scenario = st.selectbox(
            "Choose a scenario:",
            ["Custom", "Weak Confounding", "Moderate Confounding", "Strong Confounding", "Extreme Confounding"],
            key="scenario_select",
            help="Scenario sets defaults for sliders; choose Custom to edit and save your own values."
        )
        preset_map = {
            "Weak Confounding": (0.01, 0.01, 1.0, 0.95, 0.0),
            "Moderate Confounding": (0.05, 0.05, 1.0, 0.95, 0.0),
            "Strong Confounding": (0.10, 0.10, 1.0, 0.95, 0.0),
            "Extreme Confounding": (0.20, 0.20, 1.0, 0.95, 0.0),
        }
        if scenario == "Custom":
            default_cf_y, default_cf_d, default_rho, default_level, default_null = (
                st.session_state['last_cf_y'], st.session_state['last_cf_d'],
                st.session_state['last_rho'], st.session_state['last_level'],
                st.session_state['last_null_hyp']
            )
        else:
            default_cf_y, default_cf_d, default_rho, default_level, default_null = preset_map[scenario]

        # ============= 2) Parameters (after scenario) =============
        st.subheader("🎛️ Parameters")
        col1, col2 = st.columns(2)
        with col1:
            cf_y = st.number_input("Partial R² with outcome (cf_y)",
                                   min_value=0.0, max_value=1.0, value=float(default_cf_y),
                                   step=0.01, format="%.4f", key=f"cf_y_input_{scenario}")
            st.caption("Interpretation: confounder strength on outcome residual.")
        with col2:
            cf_d = st.number_input("Partial R² with treatment (cf_d)",
                                   min_value=0.0, max_value=1.0, value=float(default_cf_d),
                                   step=0.01, format="%.4f", key=f"cf_d_input_{scenario}")
            st.caption("Interpretation: confounder strength on treatment residual.")
        col3, col4 = st.columns(2)
        with col3:
            rho = st.slider("Correlation (ρ) between confounding effects",
                            min_value=-1.0, max_value=1.0, value=float(default_rho),
                            step=0.1, key=f"rho_input_{scenario}")
        with col4:
            level = st.slider("Confidence Level",
                              min_value=0.80, max_value=0.99, value=float(default_level),
                              step=0.01, format="%.2f", key=f"level_input_{scenario}")
        st.subheader("Null hypothesis (for RV / RVa)")
        null_hyp = st.number_input("H₀ (null hypothesis for θ)", value=float(default_null),
                                   step=0.1, format="%.4f", key=f"null_input_{scenario}")

        if scenario == "Custom":
            st.session_state.update(last_cf_y=cf_y, last_cf_d=cf_d, last_rho=rho,
                                    last_level=level, last_null_hyp=null_hyp)

        # ============= 3) Run =============
        st.markdown("---")
        run_col1, run_col2 = st.columns([3, 2])
        with run_col1:
            run_clicked = st.button("🔍 Run Sensitivity Analysis", type="primary", use_container_width=True)
        with run_col2:
            st.caption(f"Parameters → Scenario: **{scenario}** | cf_y: **{cf_y:.4f}** | cf_d: **{cf_d:.4f}** | "
                       f"ρ: **{rho:.2f}** | Level: **{level:.2f}** | H₀: **{null_hyp:.4f}**")

        if run_clicked:
            with st.spinner("Running sensitivity analysis..."):
                try:
                    # keep a simple run log for auditability
                    st.session_state.setdefault('sensitivity_runs', []).append({
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "scenario": scenario, "cf_y": float(cf_y), "cf_d": float(cf_d),
                        "rho": float(rho), "level": float(level), "null_hyp": float(null_hyp)
                    })

                    # Run DoubleML sensitivity
                    dml_plr.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho,
                                                 level=level, null_hypothesis=null_hyp)
                    st.success("✅ Sensitivity analysis complete!")

                    # ----- Summary -----
                    st.markdown("---")
                    st.subheader("📊 Sensitivity Summary")
                    try:
                        st.text(str(dml_plr.sensitivity_summary))
                    except Exception:
                        st.info("Summary text unavailable from backend.")

                    # ----- Plots (with scenario marker) -----
                    st.markdown("---")
                    st.subheader("📈 Sensitivity Plots")
                    tab1, tab2 = st.tabs(["Effect Bounds (θ)", "Confidence Interval Bounds"])

                    # EXACT tutorial format: benchmarks is a dict of lists
                    scenario_label = scenario if scenario != "Custom" else "Custom Scenario"
                    bench_name = f"{scenario_label} (ρ={rho:.2f})"
                    scenario_bench = {"cf_y": [float(cf_y)], "cf_d": [float(cf_d)], "name": [bench_name]}

                    with tab1:
                        plt.close('all')
                        fig_theta = dml_plr.sensitivity_plot(value="theta", benchmarks=scenario_bench)
                        title1 = f"{treatment_label}: θ bounds under {scenario} (ρ={rho:.2f})"
                        style_and_render(fig_theta, title_text=title1)
                        st.caption("Marker indicates the current scenario settings.")

                    with tab2:
                        plt.close('all')
                        fig_ci = dml_plr.sensitivity_plot(value="ci", level=level, benchmarks=scenario_bench)
                        title2 = f"{treatment_label}: {int(level*100)}% CI bounds under {scenario} (ρ={rho:.2f})"
                        style_and_render(fig_ci, title_text=title2)
                        st.caption("Where the lower bound crosses H₀ indicates required confounding to nullify the result.")

                    with st.expander("🧾 Recent sensitivity runs (most recent last)"):
                        st.dataframe(pd.DataFrame(st.session_state.sensitivity_runs).tail(10),
                                     use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Sensitivity analysis failed: {e}")
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong> DoubleML Causal Analysis Tool</strong></p>
        <p>Built by Growth by Science | Powered by DoubleML & FLAML</p>
    </div>
    """,
    unsafe_allow_html=True
)
