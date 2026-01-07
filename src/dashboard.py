import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# --- FIX: IMPORT METRICS ---
from sklearn.metrics import r2_score, mean_absolute_error

# Import functions from your modular files
from preprocessing import load_and_preprocess, get_training_data, calculate_historical_totals
from visualization import create_bar_chart, create_pie_chart, create_predictions_bar_chart, create_feature_importance_chart
from preprocessing import COL_VIEWS, COL_FILM, COL_RELEASE, COL_VIEW_MONTH_STR, COL_CATEGORY, COL_LANGUAGE

# ------------------------- Configuration / Constants -------------------------
APP_TITLE = "IMovie — December 2025 Marketing Strategy"
MODEL_CACHE_PATH = "imovie_rf_model.joblib"

# ------------------------- Modeling Utilities -------------------------
@st.cache_resource(show_spinner="Training advanced Random Forest Regressor...")
def train_model(df_model):
    X_train, y_train, X_test, y_test = get_training_data(df_model)
    
    # Base model and tuning grid
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    
    search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate
    r2 = best_model.score(X_test, y_test)
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Save model and features
    joblib.dump({'model': best_model, 'feature_columns': X_train.columns.tolist()}, MODEL_CACHE_PATH)

    return {
        'model': best_model,
        'feature_columns': X_train.columns.tolist(),
        'metrics': {'r2': r2, 'mae': mae}
    }

# ------------------------- App Layout & Execution -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.write("Data-driven strategy based on historical performance (up to Nov 2025) and future predictions.")

    # 1. Load and preprocess data
    df_model, df_full = load_and_preprocess()

    if df_model is None:
        return

    st.markdown("---")
    
    # 2. Train or load model
    model_info = {}
    try:
        if os.path.exists(MODEL_CACHE_PATH):
            model_cache = joblib.load(MODEL_CACHE_PATH)
            model_info = {'model': model_cache['model'], 'feature_columns': model_cache['feature_columns']}
            st.info("Loaded cached predictive model.")
        else:
            model_info = train_model(df_model)
            st.success("Model trained and cached.")
        
        st.subheader("Model Performance & Validation")
        st.write(f"R² Score on Test Data: **{model_info['metrics']['r2']:.4f}** (Explaining {model_info['metrics']['r2']:.2%} of variance)")
        st.write(f"Mean Absolute Error (MAE): **{model_info['metrics']['mae']:,.0f}** Views")
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Prediction Error: Could not train or load model. Ensure 'scikit-learn' and 'joblib' are installed. Error: {e}")
        model_info = None

    df_categories, df_languages = calculate_historical_totals(df_full[df_full[COL_VIEW_MONTH_STR] < '2025-12-01'])

    # --- 3. HISTORICAL & DESCRIPTIVE VISUALIZATIONS ---
    st.header("1. Historical Content Demand & Audience Targeting")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("##### Top View-Driving Categories")
        fig_cat = create_bar_chart(df_categories, 'Category_original', 'Total_Views', 'Total Views by Category')
        st.plotly_chart(fig_cat, use_container_width=True)

    with colB:
        st.markdown("##### Audience Reach by Language")
        fig_lang = create_pie_chart(df_languages, 'Language_original', 'Total_Views', 'Language Distribution')
        st.plotly_chart(fig_lang, use_container_width=True)
    
    st.markdown("---")


    # --- 4. PREDICTIVE INSIGHTS FOR DECEMBER 2025 ---
    if model_info:
        st.header("2. December 2025 Promotional Targets (Predicted)")
        
        full_model = pd.get_dummies(df_full, columns=[COL_CATEGORY, COL_LANGUAGE], drop_first=True)
        december_rows = full_model[full_model['Month_Number'] == 12].copy()

        if not december_rows.empty:
            X_dec = december_rows.reindex(columns=model_info['feature_columns'], fill_value=0)
            december_rows['Predicted_Views'] = model_info['model'].predict(X_dec)
            
            # Rank the predictions
            top_dec = december_rows.sort_values('Predicted_Views', ascending=False).head(10)
            
            st.write("🎯 **Top 10 Films (Predicted Views) for Direct Promotion in December:**")
            st.dataframe(top_dec[[COL_FILM, 'Language_original', 'Category_original', COL_RELEASE, 'Predicted_Views']], use_container_width=True)

            st.plotly_chart(create_predictions_bar_chart(top_dec, 'Top Predicted Films for December'), use_container_width=True)

        # Feature Importance
        st.header("3. Predictive Feature Importance")
        st.write("Factors driving the predictive model (identifies high-impact movie attributes):")
        st.plotly_chart(create_feature_importance_chart(model_info['model'], model_info['feature_columns']), use_container_width=True)


if __name__ == '__main__':
    main()