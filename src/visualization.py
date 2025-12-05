import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from preprocessing import COL_FILM, COL_LANGUAGE, COL_CATEGORY, COL_VIEWS

def create_bar_chart(df, x_col, y_col, title, limit=10, orientation='h', color_scale='Teal'):
    """Creates a Plotly Bar Chart for ranking categories or languages."""
    df_top = df.head(limit).sort_values(y_col, ascending=orientation == 'h')
    
    fig = px.bar(df_top, x=y_col, y=x_col, orientation=orientation,
                 title=title, color=y_col,
                 labels={x_col: x_col, y_col: 'Total Views (Millions)'},
                 color_continuous_scale=getattr(px.colors.sequential, color_scale))
    fig.update_layout(showlegend=False, title_x=0.5)
    fig.update_xaxes(tickformat=".2s")
    return fig

def create_pie_chart(df, names_col, values_col, title, color_scale='Teal'):
    """Creates a Plotly Pie Chart for distribution (e.g., Languages)."""
    fig = px.pie(df, names=names_col, values=values_col, 
                 title=title, hole=.3, 
                 color_discrete_sequence=getattr(px.colors.sequential, color_scale))
    fig.update_layout(title_x=0.5)
    return fig

def create_predictions_bar_chart(df, title, color_scale='Teal'):
    """Creates a horizontal bar chart for the top predicted movies."""
    df['Predicted_Views'] = df['Predicted_Views'].round(0).astype(int)
    
    fig = px.bar(df, x='Predicted_Views', y=COL_FILM, orientation='h', 
                     title=title, color='Predicted_Views', 
                     hover_data=['Category_original', 'Language_original'],
                     labels={COL_FILM: 'Film Entity', 'Predicted_Views': 'Predicted Views (Dec 2025)'},
                     color_continuous_scale=getattr(px.colors.sequential, color_scale))
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_xaxes(tickformat=".2s")
    fig.update_layout(showlegend=False, title_x=0.5)
    return fig

def create_feature_importance_chart(model, feature_names):
    """Creates a chart showing the importance of features in the ML model."""
    importances = model.feature_importances_
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(15)

    fig = px.bar(fi, x='importance', y='feature', orientation='h', 
                 title='Top 15 Feature Importances in Prediction Model',
                 color='importance', color_continuous_scale=px.colors.sequential.Teal)
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(showlegend=False, title_x=0.5)
    return fig