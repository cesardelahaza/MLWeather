import plotly.express as px

def line_chart(df, y_col, title, y_label):
    return px.line(df, x='date', y=y_col,
                   title=title, color='d_type',
                   color_discrete_map={"Historic": "blue", "Forecast": "red"},
                   labels={'date': 'Date', 
                           y_col: y_label, 
                           'd_type': ''})

def model_predictions(df, model_name):
    fig = px.line(df, x='date', y=['target', 'y_pred'], 
                  title= model_name + ' predictions',
                  labels={'date': 'Date',
                          'value': 'Temperature',
                          'variable': ''})
    
    fig.for_each_trace(lambda trace: trace.update(name=trace.name.replace("target", "Real").replace("y_pred", "Predicted")))
    
    return fig

def most_imp_feats(importances_with_names, model):
    return px.bar(x=importances_with_names.values,
                  y=importances_with_names.index,
                  orientation='h',
                  title=f'Top 15 most important coefficients ({model})',
                  labels={'x': 'Importance', 'y': 'Feature'}
                  )

def pred_vs_real(model_data, model):
    fig = px.scatter(model_data, x='target', y='y_pred',
                     title=f'Predicted vs Real temperature ({model})',
                     labels={'target': 'Real (°C)', 
                             'y_pred': 'Predicted (°C)'})

    fig.add_shape(type='line',
                  x0=model_data['target'].min(),
                  y0=model_data['target'].min(),
                  x1=model_data['target'].max(),
                  y1=model_data['target'].max(),
                  line=dict(color='red', dash='dash'))
    
    return fig

def residuals(model_data, model):
    fig = px.line(model_data, x='date', y='residuals',
                  title=f'Residuals ({model})',
                  labels={'date': 'Date',
                          'residuals': 'Residuals'})
    
    fig.add_hline(y=0, line=dict(color='red', dash='dash'))

    return fig