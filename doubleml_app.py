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
                    
                    # Prepare DoubleML
                    y_for_doubleml = y_outcome.iloc[mask]
                    d_for_doubleml = d_treatment.iloc[mask]
                    
                    df_for_doubleml = pd.concat([
                        pd.Series(y_for_doubleml, name=config['outcome_col'], index=y_for_doubleml.index),
                        pd.Series(d_for_doubleml, name=treatment_var_name, index=y_for_doubleml.index),
                        pd.Series(np.zeros_like(y_for_doubleml), name="dummy_x", index=y_for_doubleml.index),
                    ], axis=1)
                    
                    dml_data = dml.DoubleMLData(
                        df_for_doubleml, 
                        y_col=config['outcome_col'], 
                        d_cols=treatment_var_name, 
                        x_cols=["dummy_x"]
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
                    
                    # Store results
                    st.session_state.results = {
                        'dml_plr': dml_plr,
                        'cv_loss_l': cv_loss_l,
                        'cv_loss_m': cv_loss_m,
                        'n_obs': dml_data.n_obs,
                        'mask_sum': mask.sum(),
                        'total_obs': len(df_model),
                        'confounder_cols': config['confounder_cols'],
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
    
    # Check for the correct session state key
    if st.session_state.get('results') is None or st.session_state.results.get('dml_plr') is None:
        st.warning("⚠️ Please run the causal analysis first (Step 3)")
        st.info("The sensitivity analysis tests how robust your results are to **unobserved confounding**.")
    else:
        # Assuming the dml_plr object is stored in st.session_state.results['dml_plr']
        dml_plr = st.session_state.results['dml_plr']

         # Ensure a treatment name exists (prevents 'undefined' in titles/labels)
        treatment_from_app = st.session_state.results.get('treatment_var_name', None)
        if getattr(dml_plr, "d_cols", None) and len(dml_plr.d_cols) > 0:
            treatment_names = list(dml_plr.d_cols)
        else:
            treatment_names = [treatment_from_app or "treatment"]
            dml_plr.d_cols = treatment_names  # for plotting/title consistency

        # --- Helper: render whatever DoubleML returns (Plotly or Matplotlib) ---
        try:
            import plotly.graph_objs as go
            _PlotlyFigure = go.Figure
        except Exception:
            _PlotlyFigure = tuple()  # plotly not available

        from matplotlib.figure import Figure as _MplFigure
        from matplotlib.axes import Axes as _MplAxes
        import matplotlib.pyplot as plt

        def _render_doubleml_plot(plot_obj):
            # Plotly Figure?
            if _PlotlyFigure and isinstance(plot_obj, _PlotlyFigure):
                st.plotly_chart(plot_obj, use_container_width=True)
                return
            # Matplotlib Figure?
            if isinstance(plot_obj, _MplFigure) or hasattr(plot_obj, "savefig"):
                fig = plot_obj
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig)
                plt.close(fig)
                return
            # Matplotlib Axes?
            if isinstance(plot_obj, _MplAxes) and hasattr(plot_obj, "figure"):
                fig = plot_obj.figure
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig)
                plt.close(fig)
                return
            # None (drawn on current figure)?
            fig = plt.gcf()
            if isinstance(fig, _MplFigure):
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Plot object not recognized; nothing to display.")

        st.info("""
        📊 **Sensitivity Analysis** assesses how robust your causal estimates are to potential **unobserved confounding**.
        
        Even after controlling for observed confounders, there may be unmeasured variables that affect both treatment and outcome.
        This analysis shows how strong such confounding would need to be to change your conclusions.
        """)
        
        # Sensitivity Parameters
        st.subheader("🎛️ Sensitivity Parameters")
        
        # Initialize default values in a robust way
        default_cf_y = st.session_state.get('cf_y', 0.05)
        default_cf_d = st.session_state.get('cf_d', 0.05)
        default_rho = st.session_state.get('rho', 1.0)
        default_level = st.session_state.get('level', 0.95)

        col1, col2 = st.columns(2)
        
        with col1:
            cf_y = st.number_input(
                "Partial R² with outcome (cf_y)",
                min_value=0.0,
                max_value=1.0,
                value=default_cf_y,
                step=0.01,
                format="%.4f",
                help="Strength of association between unobserved confounder and outcome (0-1)"
            )
            st.caption("💡 Represents how much of the outcome variance the unobserved confounder explains")
        
        with col2:
            cf_d = st.number_input(
                "Partial R² with treatment (cf_d)",
                min_value=0.0,
                max_value=1.0,
                value=default_cf_d,
                step=0.01,
                format="%.4f",
                help="Strength of association between unobserved confounder and treatment (0-1)"
            )
            st.caption("💡 Represents how much of the treatment variance the unobserved confounder explains")
        
        col3, col4 = st.columns(2)
        
        with col3:
            rho = st.slider(
                "Correlation (ρ)",
                min_value=-1.0,
                max_value=1.0,
                value=default_rho,
                step=0.1,
                help="Correlation between confounding effects on outcome and treatment"
            )
        
        with col4:
            level = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=default_level,
                step=0.01,
                format="%.2f"
            )
        
        # Store current values for persistence
        st.session_state.cf_y = cf_y
        st.session_state.cf_d = cf_d
        st.session_state.rho = rho
        st.session_state.level = level

        # Preset scenarios
        st.subheader("📋 Preset Scenarios")
        scenario = st.selectbox(
            "Or choose a preset scenario:",
            ["Custom", "Weak Confounding", "Moderate Confounding", "Strong Confounding", "Extreme Confounding"]
        )
        if scenario == "Weak Confounding":
            cf_y, cf_d = 0.01, 0.01
        elif scenario == "Moderate Confounding":
            cf_y, cf_d = 0.05, 0.05
        elif scenario == "Strong Confounding":
            cf_y, cf_d = 0.10, 0.10
        elif scenario == "Extreme Confounding":
            cf_y, cf_d = 0.20, 0.20
        
        # Benchmarking
        st.markdown("---")
        st.subheader("📊 Benchmarking Against Observed Variables")
        st.write("""
        Compare the sensitivity parameters to observed confounders to understand what level of 
        unobserved confounding would be needed to overturn your results.
        """)
        available_vars = st.session_state.results.get('confounder_cols', [])
        if available_vars:
            default_benchmarks = st.session_state.get('benchmark_vars', available_vars[:min(3, len(available_vars))])
            benchmark_vars = st.multiselect(
                "Select variables to use as benchmarks:",
                available_vars,
                default=default_benchmarks,
                help="These variables will be used to calibrate the sensitivity analysis"
            )
            st.session_state.benchmark_vars = benchmark_vars
        else:
            st.warning("⚠️ No confounding variables available for benchmarking")
            benchmark_vars = []
        
        # Run Analysis Button
        st.markdown("---")
        if st.button("🔍 Run Sensitivity Analysis", type="primary", use_container_width=True):
            with st.spinner("Running sensitivity analysis..."):
                try:
                    # Run sensitivity analysis for chosen parameters
                    dml_plr.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho, level=level)
                    st.success("✅ Sensitivity analysis complete!")
                    
                    # --- Summary: Build a DataFrame from sensitivity_params (version-proof) ---
                    st.markdown("---")
                    st.subheader("📊 Sensitivity Summary")

                    params = getattr(dml_plr, "sensitivity_params", None)
                    if params is not None:
                        try:
                            tname = dml_plr.d_cols[0] if hasattr(dml_plr, "d_cols") else "treatment"
                        except Exception:
                            tname = "treatment"

                        row = {
                            "theta lower": float(params["theta"]["lower"][0]),
                            "theta":       float(getattr(dml_plr, "coef", [np.nan])[0]),
                            "theta upper": float(params["theta"]["upper"][0]),
                            "CI lower":    float(params["ci"]["lower"][0]),
                            "CI upper":    float(params["ci"]["upper"][0]),
                            "RV (%)":      float(params.get("rv", [np.nan])[0]) * 100 if params.get("rv") is not None else np.nan,
                            "RVa (%)":     float(params.get("rva", [np.nan])[0]) * 100 if params.get("rva") is not None else np.nan,
                        }
                        summary_df = pd.DataFrame([row], index=[tname])
                        st.dataframe(summary_df, use_container_width=True)
                    else:
                        # Fallback: print whatever summary exists (string or DataFrame)
                        summary = getattr(dml_plr, "sensitivity_summary", None)
                        if isinstance(summary, pd.DataFrame):
                            st.dataframe(summary, use_container_width=True)
                        elif summary is not None:
                            st.markdown(f"```text\n{summary}\n```")
                        else:
                            st.info("No sensitivity summary was returned by DoubleML.")

                    # --- Plots: Handle Plotly/Mpl/None robustly ---
                    st.markdown("---")
                    st.subheader("📈 Sensitivity Plots")
                    tab1, tab2 = st.tabs(["Effect Bounds (θ)", "Confidence Interval Bounds"])
                    
                    with tab1:
                        st.write("Shows how the **point estimate** changes with different levels of confounding:")
                        plt.close('all')
                        obj_theta = dml_plr.sensitivity_plot(value='theta')
                        _render_doubleml_plot(obj_theta)
                        st.caption("The plot shows the estimated causal effect across different values of confounding strength. The shaded region represents the sensitivity bounds.")
                    
                    with tab2:
                        st.write(f"Shows how the **{level*100:.0f}% confidence interval** changes with different levels of confounding:")
                        plt.close('all')
                        obj_ci = dml_plr.sensitivity_plot(value='ci', level=level)
                        _render_doubleml_plot(obj_ci)
                        st.caption("The plot shows the confidence interval bounds across different values of confounding strength. The line where the lower bound crosses zero indicates the confounding needed to make the effect insignificant.")

                    # Benchmarking Results
                    if benchmark_vars:
                        st.markdown("---")
                        st.subheader("🎯 Benchmarking Results")
                        st.write("""
                        These results compare the sensitivity parameters to observed confounders,
                        helping you understand what "strength" of unobserved confounding would be needed.
                        """)
                        for var in benchmark_vars:
                            with st.expander(f"📊 Benchmark: **{var}**"):
                                try:
                                    bench_result = dml_plr.sensitivity_benchmark(benchmarking_set=[var])
                                    if hasattr(bench_result, 'to_frame'):
                                        st.dataframe(bench_result.to_frame().T, use_container_width=True)
                                    elif isinstance(bench_result, dict):
                                        st.write("**Benchmark Parameters:**")
                                        for key, value in bench_result.items():
                                            try:
                                                st.write(f"- **{key}**: `{float(value):.4f}`")
                                            except Exception:
                                                st.write(f"- **{key}**: `{value}`")
                                    else:
                                        st.write(bench_result)
                                    st.info(f"""
                                    This shows the confounding strength of **{var}**. 
                                    If unobserved confounding is similar in strength to **{var}**, 
                                    these would be the sensitivity parameters (cf_y and cf_d).
                                    """)
                                except Exception as e:
                                    st.error(f"Error benchmarking {var}: {str(e)}")
                    
                    # Additional Insights
                    st.markdown("---")
                    st.subheader("💡 Key Insights")
                    with st.expander("🤔 How to interpret these results"):
                        st.markdown("""
                        **Sensitivity Analysis Guidelines:**
                        
                        1. **Small cf_y and cf_d values (< 0.05)**: Your results are robust to weak unobserved confounding
                        2. **Moderate values (0.05 - 0.15)**: Results require careful interpretation
                        3. **Large values (> 0.15)**: Results are sensitive to unobserved confounding
                        
                        **Benchmarking helps you answer:**
                        - "How strong would an unobserved confounder need to be compared to my observed confounders?"
                        - "Is it plausible that such a confounder exists in my context?"
                        
                        **Best practices:**
                        - Compare cf_y and cf_d to R² values from your observed confounders
                        - Consider domain knowledge about potential unobserved confounders
                        - Report sensitivity results alongside main estimates
                        - If results are sensitive, consider collecting additional covariates
                        """)
                    
                    with st.expander("📚 Learn More"):
                        st.markdown("""
                        **Resources:**
                        - [DoubleML Documentation](https://docs.doubleml.org/)
                        - Chernozhukov et al. (2018): "Double/debiased machine learning for treatment and structural parameters"
                        - Cinelli & Hazlett (2020): "Making sense of sensitivity: Extending omitted variable bias"
                        
                        **Citation:**
                        If you use this tool in research, please cite the DoubleML package and relevant methodology papers.
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Error during sensitivity analysis: {str(e)}")
                    st.exception(e)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>Generic DoubleML Causal Analysis Tool</strong></p>
        <p>Built by Growth by Science | Powered by DoubleML & FLAML</p>
        <p style='font-size: 0.9em;'>Support for Marketing, Healthcare, Policy, Finance & More</p>
    </div>
    """,
    unsafe_allow_html=True
)
