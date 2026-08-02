"""
The Wage vs Price Divergence — Streamlit App
=============================================

QUICK START:
    pip install streamlit pandas plotly scikit-learn
    streamlit run app.py

Single file. Data is embedded below. No CSVs to upload.
"""
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

st.set_page_config(page_title="Wage vs Price Divergence", page_icon="👕", layout="wide")

# ---------- EMBEDDED DATA ----------
DATA = """Year,Apparel_Idx,Overall_Idx,Wage_Idx,Nominal_Wage,Real_Wage,BD_CPI,BD_Inflation,Real_Wage_YoY,Years_Since_Revision,Protected
1990,95.76,75.88,112.7,627.0,1944.19,32.25,,,5,
1991,99.36,79.08,105.96,627.0,1827.99,34.3,6.36,-5.98,6,0
1992,101.79,81.48,102.24,627.0,1763.71,35.55,3.64,-3.52,7,0
1993,103.16,83.9,99.25,627.0,1712.18,36.62,3.01,-2.92,8,0
1994,102.97,86.08,139.81,930.0,2411.83,38.56,5.3,40.86,0,1
1995,101.86,88.5,126.76,930.0,2186.69,42.53,10.3,-9.33,1,0
1996,101.65,91.1,123.79,930.0,2135.48,43.55,2.4,-2.34,2,0
1997,102.53,93.22,117.55,930.0,2027.91,45.86,5.3,-5.04,3,0
1998,102.65,94.67,108.45,930.0,1870.85,49.71,8.4,-7.74,4,0
1999,101.29,96.74,102.22,930.0,1763.37,52.74,6.1,-5.75,5,0
2000,100.0,100.0,100.0,930.0,1725.1,53.91,2.22,-2.17,6,0
2001,98.21,102.82,98.04,930.0,1691.22,54.99,2.0,-1.96,7,0
2002,95.68,104.46,94.88,930.0,1636.75,56.82,3.33,-3.22,8,0
2003,93.27,106.86,89.79,930.0,1548.97,60.04,5.67,-5.36,9,0
2004,92.95,109.71,83.45,930.0,1439.63,64.6,7.59,-7.06,10,0
2005,92.24,113.4,77.96,930.0,1344.9,69.15,7.04,-6.58,11,0
2006,92.18,117.05,130.49,1662.0,2251.12,73.83,6.77,67.38,0,1
2007,91.82,120.41,119.59,1662.0,2063.06,80.56,9.12,-8.35,1,0
2008,91.76,125.01,109.82,1662.0,1894.45,87.73,8.9,-8.17,2,0
2009,92.65,124.61,104.18,1662.0,1797.15,92.48,5.41,-5.14,3,0
2010,92.21,126.65,173.9,3000.0,3000.0,100.0,8.13,66.93,0,1
2011,94.22,130.62,156.11,3000.0,2693.0,111.4,11.4,-10.23,1,0
2012,97.43,133.33,146.98,3000.0,2535.5,118.32,6.21,-5.85,2,0
2013,98.34,135.29,241.48,5300.0,4165.68,127.23,7.53,64.29,0,1
2014,98.43,137.47,225.69,5300.0,3893.34,136.13,7.0,-6.54,1,0
2015,97.2,137.64,212.53,5300.0,3666.3,144.56,6.19,-5.83,2,0
2016,97.28,139.38,201.42,5300.0,3474.73,152.53,5.51,-5.23,3,0
2017,96.96,142.35,190.55,5300.0,3287.23,161.23,5.7,-5.4,4,0
2018,96.98,145.83,272.53,8000.0,4701.46,170.16,5.54,43.02,0,1
2019,95.76,148.47,258.09,8000.0,4452.36,179.68,5.59,-5.3,1,0
2020,91.13,150.33,244.19,8000.0,4212.52,189.91,5.69,-5.39,2,0
2021,93.4,157.37,231.36,8000.0,3991.22,200.44,5.54,-5.25,3,0
2022,98.08,169.94,214.83,8000.0,3706.11,215.86,7.69,-7.14,4,0
2023,100.77,176.96,305.48,12500.0,5269.81,237.2,9.89,42.19,0,1
2024,101.46,182.18,276.54,12500.0,4770.63,262.02,10.46,-9.47,1,0
2025,101.43,186.98,254.24,12500.0,4385.96,285.0,8.77,-8.06,2,0
2026,104.57,191.97,,,,,,,3,"""

@st.cache_data
def load():
    return pd.read_csv(io.StringIO(DATA))

df = load()

# ---------- HEADER ----------
st.title("👕 Fast Fashion: The Wage vs Price Divergence")
st.caption("How US clothing stayed cheap while everything else — including workers' living costs — got expensive")

# ---------- HEADLINE METRICS ----------
latest = df[df['Wage_Idx'].notna()].iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("US Apparel Prices", f"+{latest['Apparel_Idx']-100:.0f}%", "cumulative since 2000")
c2.metric("US Overall CPI",    f"+{latest['Overall_Idx']-100:.0f}%", "cumulative since 2000")
c3.metric("Years wages fell",  f"{(df['Protected']==0).sum()} / {df['Protected'].notna().sum()}",
          "since 1990")

# ---------- TABS ----------
t1, t2, t3, t4 = st.tabs(["🎯 The Divergence", "📉 Wage Freeze",
                          "🤖 Regression Forecast", "🌳 Decision Tree"])

# ---------- TAB 1: DIVERGENCE ----------
with t1:
    st.subheader("The Divergence")
    long = df.melt(id_vars='Year',
                   value_vars=['Apparel_Idx','Overall_Idx','Wage_Idx'],
                   var_name='Series', value_name='Index').dropna()
    long['Series'] = long['Series'].map({
        'Apparel_Idx':'US Apparel Prices',
        'Overall_Idx':'US Overall CPI',
        'Wage_Idx':'BD Real Wage',
    })
    fig = px.line(long, x='Year', y='Index', color='Series',
                  color_discrete_map={
                      'US Apparel Prices':'#c0392b',
                      'US Overall CPI':'#2c3e50',
                      'BD Real Wage':'#e67e22',
                  },
                  title="All series indexed to 2000 = 100")
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("US apparel barely rose while overall CPI nearly doubled. Someone absorbed the cost.")

# ---------- TAB 2: WAGE FREEZE ----------
with t2:
    st.subheader("Bangladesh RMG Wage — Nominal vs Real")
    d = df.dropna(subset=['Nominal_Wage'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['Year'], y=d['Nominal_Wage'], name='Nominal (BDT)',
                             line=dict(color='#27ae60', width=3, shape='hv')))
    fig.add_trace(go.Scatter(x=d['Year'], y=d['Real_Wage'], name='Real (2010 BDT)',
                             yaxis='y2',
                             line=dict(color='#c0392b', width=2.5, dash='dash')))
    fig.update_layout(
        xaxis_title='Year',
        yaxis=dict(title='Nominal (BDT)', color='#27ae60'),
        yaxis2=dict(title='Real (2010 BDT)', overlaying='y', side='right', color='#c0392b'),
        height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nominal jumps at every revision. Real purchasing power erodes in between.")

# ---------- TAB 3: LINEAR REGRESSION ----------
with t3:
    st.subheader("Linear Regression Forecast to 2030")
    fig = go.Figure()
    palette = {'Apparel_Idx':'#c0392b','Overall_Idx':'#2c3e50','Wage_Idx':'#e67e22'}
    labels  = {'Apparel_Idx':'US Apparel','Overall_Idx':'US Overall CPI','Wage_Idx':'BD Real Wage'}

    rows = []
    for col in ['Apparel_Idx','Overall_Idx','Wage_Idx']:
        d = df.dropna(subset=[col])
        X = d[['Year']].values; y = d[col].values
        m = LinearRegression().fit(X, y)
        future = np.arange(1990, 2031).reshape(-1, 1)
        preds = m.predict(future)
        r2 = m.score(X, y)

        fig.add_trace(go.Scatter(x=d['Year'], y=y, mode='markers',
                                 name=labels[col], marker=dict(color=palette[col])))
        fig.add_trace(go.Scatter(x=future.ravel(), y=preds, mode='lines',
                                 line=dict(color=palette[col]), showlegend=False))
        rows.append({'Series': labels[col], 'Slope': round(m.coef_[0], 2),
                      'R²': round(r2, 3), '2030 forecast': round(preds[-1], 1)})

    fig.add_vline(x=int(df[df['Wage_Idx'].notna()]['Year'].max()),
                  line_dash="dash", line_color="gray")
    fig.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("The gap between US Apparel and US Overall CPI widens through 2030 — structural, not cyclical.")

# ---------- TAB 4: DECISION TREE ----------
with t4:
    st.subheader("Decision Tree Classification — What predicts wage protection?")
    st.markdown("**Target:** Was the year Protected (wage grew) or Eroded (wage fell)?")

    # Prepare features
    clf = df.dropna(subset=['Protected','BD_Inflation','Nominal_Wage','Real_Wage_YoY']).copy()
    features = ['Years_Since_Revision', 'BD_Inflation', 'Nominal_Wage']
    X = clf[features]; y = clf['Protected'].astype(int)

    # GridSearchCV
    param_grid = {'max_depth':[2,3,4,5,None], 'min_samples_split':[2,3,4],
                  'criterion':['gini','entropy']}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                        param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid.fit(X, y)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Best CV F1 score", f"{grid.best_score_:.3f}")
        st.write("**Best hyperparameters:**")
        st.json(grid.best_params_)

    with c2:
        imp = pd.DataFrame({
            'Feature': features,
            'Importance': grid.best_estimator_.feature_importances_.round(3)
        }).sort_values('Importance', ascending=False)
        st.write("**Feature importance:**")
        st.dataframe(imp, hide_index=True, use_container_width=True)

    fig = px.bar(imp, x='Feature', y='Importance',
                 color='Importance', color_continuous_scale=['#e0e0e0','#c0392b'],
                 title='Which feature drives wage protection?')
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "**Finding:** `Years_Since_Revision` alone perfectly predicts wage protection. "
        "Inflation and wage level score 0.00 — they don't matter. "
        "Wages are a POLITICAL variable, not an economic one."
    )

# ---------- FOOTER ----------
st.divider()
st.caption(
    "**Data:** US CPI (FRED / BLS), Bangladesh CPI (World Bank), "
    "wage revisions (Bangladesh Minimum Wage Board). "
    "MSc Applied Finance & Wealth Management project."
)
