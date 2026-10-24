import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="AutoMPG • AI Garage", page_icon="🚘", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def reset_app():
    for k in list(st.session_state.keys()):
        if k.startswith("veh_") or k in ["results", "best_name", "pipeline", "feature_columns", "training_df", "prediction_done", "last_prediction"]:
            del st.session_state[k]

dark = st.session_state.dark_mode
if dark:
    bg, panel, panel2, fg, muted, border, accent, accent2 = "#080d19", "#111a2b", "#17233a", "#edf4ff", "#9aabc6", "#263754", "#6d7cff", "#21d4fd"
else:
    bg, panel, panel2, fg, muted, border, accent, accent2 = "#f3f6fc", "#ffffff", "#eef3ff", "#17233b", "#64748b", "#dce5f4", "#4f46e5", "#0891b2"

st.markdown(f"""
<style>
.stApp{{background:radial-gradient(ellipse at 8% 0%,rgba(65,90,255,.15),transparent 35%),radial-gradient(ellipse at 92% 12%,rgba(33,212,253,.11),transparent 32%),{bg};color:{fg}}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:{panel};border-right:1px solid {border}}}
[data-testid="stSidebar"] *{{color:{fg}}}
.block-container{{padding-top:1.4rem;max-width:1450px}}
h1,h2,h3,h4,p,label,.stMarkdown{{color:{fg}}}
.hero{{position:relative;overflow:hidden;padding:30px 32px;border-radius:24px;background:linear-gradient(120deg,{panel},{panel2},{'#123a50' if dark else '#dff8ff'});border:1px solid {border};box-shadow:0 16px 42px rgba(0,0,0,.18);margin-bottom:22px;animation:rise .7s ease-out both}}
.hero:after{{content:"";position:absolute;width:270px;height:270px;border-radius:50%;right:-55px;top:-110px;background:linear-gradient(135deg,{accent},{accent2});opacity:.2;animation:glow 7s ease-in-out infinite alternate}}
.kicker{{text-transform:uppercase;letter-spacing:3px;font-size:.75rem;font-weight:800;color:{accent2}}}
.hero h1{{font-size:clamp(2rem,4vw,3.2rem);line-height:1.06;margin:.45rem 0 .65rem;font-weight:850;letter-spacing:-1.5px}}
.hero p{{color:{muted};max-width:760px;margin:0;font-size:1.02rem}}
.pill{{display:inline-block;border:1px solid {border};background:{panel2};padding:6px 11px;border-radius:999px;font-size:.78rem;margin:14px 6px 0 0}}
.section-title{{font-size:1.25rem;font-weight:800;margin:20px 0 10px}}
div[data-testid="stMetric"]{{background:{panel};border:1px solid {border};border-radius:18px;padding:16px 18px;box-shadow:0 8px 22px rgba(0,0,0,.12);transition:transform .22s ease,border-color .22s ease;animation:rise .55s ease-out both}}
div[data-testid="stMetric"]:hover{{transform:translateY(-4px);border-color:{accent}}}
div[data-testid="stMetricLabel"] p{{color:{muted}!important;font-weight:600}}
div[data-testid="stMetricValue"]{{color:{fg}!important;font-weight:800}}
div[data-testid="stForm"],div[data-testid="stExpander"]{{background:{panel};border:1px solid {border};border-radius:18px;padding:12px}}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{{border:0;border-radius:12px;padding:.68rem 1.05rem;font-weight:750;background:linear-gradient(105deg,{accent},{accent2});color:white;transition:transform .2s ease,filter .2s ease}}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{{transform:translateY(-2px);filter:brightness(1.08);color:white}}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{{background:{panel};border-color:{border};border-radius:10px}}
[data-testid="stDataFrame"]{{border:1px solid {border};border-radius:14px;overflow:hidden}}
hr{{border-color:{border}}}
@keyframes rise{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes glow{{from{{transform:translate(0,0) scale(1)}}to{{transform:translate(-28px,25px) scale(1.13)}}}}
@media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚘 AI Garage")
    st.caption("Vehicle efficiency lab")
    choice = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
    if choice != st.session_state.dark_mode:
        st.session_state.dark_mode = choice
        st.rerun()
    st.divider()
    st.markdown("### 📂 Dataset")
    source = st.radio("Data source", ["Built-in sample dataset", "Upload my CSV"])
    uploaded = st.file_uploader("Choose CSV dataset", type=["csv"]) if source == "Upload my CSV" else None
    st.caption("CSV needs an `mpg` or `Miles Per Gallon` target column.")
    st.divider()
    st.button("↻ Reset app", on_click=reset_app, use_container_width=True)
    st.caption("Clears trained models, previous predictions, and vehicle inputs.")

st.markdown("""
<div class="hero">
<div class="kicker">Machine learning · Regression studio</div>
<h1>Drive smarter.<br>Predict your mileage.</h1>
<p>Explore vehicle data, benchmark three ML models, and estimate fuel efficiency in an animated dashboard.</p>
<span class="pill">✦ 3 ML algorithms</span><span class="pill">⚡ Automated preprocessing</span><span class="pill">📈 Model evaluation</span>
</div>
""", unsafe_allow_html=True)

aliases = {
 "mpg":["mpg","miles per gallon","miles_per_gallon"], "cylinders":["cylinders","cylinder"],
 "displacement":["displacement","engine displacement"], "horsepower":["horsepower","hp"],
 "weight":["weight","vehicle weight"], "acceleration":["acceleration","accel"],
 "model_year":["model_year","model year","year"], "origin":["origin","country of origin"],
 "car_name":["car_name","car name","name","vehicle"]
}
def normalize(df):
    lookup={str(c).strip().lower():c for c in df.columns}
    return df.rename(columns={lookup[a]:s for s, al in aliases.items() for a in al if a in lookup and not any(lookup[a] == v for v in [])})

@st.cache_data
def load_data():
    local=os.path.join(os.path.dirname(__file__),"auto-mpg.csv")
    if os.path.exists(local): return pd.read_csv(local)
    return pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")

def prepare(raw):
    df=normalize(raw.copy())
    if "mpg" not in df.columns: raise ValueError("Dataset must contain mpg or Miles Per Gallon.")
    df["mpg"]=pd.to_numeric(df["mpg"],errors="coerce")
    df=df.dropna(subset=["mpg"])
    for c in df.columns:
        if c=="mpg": continue
        if c=="car_name": df[c]=df[c].fillna("Unknown").astype(str)
        elif df[c].dtype=="object":
            converted=pd.to_numeric(df[c].replace("?",pd.NA),errors="coerce")
            if converted.notna().sum() >= max(1,int(len(df)*.5)): df[c]=converted
        else: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(axis=1,how="all")
    if len(df)<10: raise ValueError("At least 10 valid rows are required.")
    return df

def train(df):
    X,y=df.drop(columns=["mpg"]),df["mpg"]
    nums=X.select_dtypes(include="number").columns.tolist()
    cats=X.select_dtypes(exclude="number").columns.tolist()
    trans=[]
    if nums: trans.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats: trans.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cats))
    prep=ColumnTransformer(trans)
    models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=250,random_state=42),"Gradient Boosting":GradientBoostingRegressor(random_state=42)}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    rows=[]; fitted={}
    for name,model in models.items():
        pipe=Pipeline([("preprocessor",prep),("model",model)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        rows.append({"Model":name,"R²":round(r2_score(yte,pred),4),"MAE":round(mean_absolute_error(yte,pred),4),"RMSE":round(mean_squared_error(yte,pred)**.5,4)})
        fitted[name]=pipe
    results=pd.DataFrame(rows).sort_values(["R²","MAE"],ascending=[False,True]).reset_index(drop=True)
    best=results.iloc[0]["Model"]
    return results,best,fitted[best],X.columns.tolist()

try:
    if source=="Upload my CSV":
        if uploaded is None:
            st.info("👈 Upload a CSV in the sidebar to begin, or switch to the built-in sample dataset.")
            st.stop()
        raw=pd.read_csv(uploaded)
    else: raw=load_data()
    df=prepare(raw)
except Exception as e:
    st.error(f"Dataset could not be loaded: {e}")
    st.stop()

st.markdown('<div class="section-title">📊 At a glance</div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
a.metric("Dataset records",f"{len(df):,}","valid MPG rows")
b.metric("Input features",str(len(df.columns)-1),"vehicle attributes")
c.metric("Average MPG",f"{df.mpg.mean():.1f}","across dataset")
d.metric("Missing cells",f"{int(df.isna().sum().sum()):,}","handled in pipeline")
with st.expander("🔎 Explore dataset preview and data quality"):
    t1,t2=st.tabs(["Dataset preview","Data quality"])
    with t1: st.dataframe(df.head(25),use_container_width=True,hide_index=True)
    with t2: st.dataframe(pd.DataFrame({"Column":df.columns,"Missing values":[int(df[x].isna().sum()) for x in df.columns],"Data type":[str(df[x].dtype) for x in df.columns]}),use_container_width=True,hide_index=True)

st.markdown('<div class="section-title">🧠 Model lab</div>',unsafe_allow_html=True)
st.write("Train three regression models on a held-out test set, then compare their performance.")
if st.button("⚡ Train & compare models",type="primary"):
    with st.spinner("Tuning up the models… analyzing vehicle patterns"):
        try:
            res,best,pipe,features=train(df)
            st.session_state["results"]=res; st.session_state["best_name"]=best
            st.session_state["pipeline"]=pipe; st.session_state["feature_columns"]=features; st.session_state["training_df"]=df
        except Exception as e: st.error(f"Training failed: {e}")

if "results" in st.session_state:
    x,y=st.columns([1.5,1])
    with x:
        st.dataframe(st.session_state["results"],use_container_width=True,hide_index=True)
        st.download_button("⬇ Download comparison CSV",st.session_state["results"].to_csv(index=False).encode(),"model_comparison.csv","text/csv")
    with y:
        st.markdown("#### 🏆 Champion model")
        st.markdown(f"### {st.session_state['best_name']}")
        st.caption("Highest test R², with MAE as tie-breaker.")
        row=st.session_state["results"].iloc[0]
        st.metric("Test R²",f"{row['R²']:.3f}")
        st.metric("Mean absolute error",f"{row['MAE']:.2f} MPG")

    st.markdown('<div class="section-title">🔮 Mileage prediction</div>',unsafe_allow_html=True)
    st.write("Enter vehicle specifications and estimate its fuel efficiency.")
    pipe=st.session_state["pipeline"]; train_df=st.session_state["training_df"]; features=st.session_state["feature_columns"]
    with st.form("prediction_form"):
        vals={}; left,right=st.columns(2)
        for i,col in enumerate(features):
            series=train_df[col]; key="veh_"+col; label=col.replace("_"," ").title()
            with (left if i%2==0 else right):
                if pd.api.types.is_numeric_dtype(series):
                    default=float(series.median()) if series.notna().any() else 0.0
                    if col=="cylinders": vals[col]=st.number_input(label,min_value=1,max_value=16,value=max(1,int(default)),key=key)
                    elif col=="origin": vals[col]=st.selectbox("Origin (1=USA, 2=Europe, 3=Japan)",[1,2,3],index=max(0,min(2,int(default)-1)),key=key)
                    elif col=="model_year": vals[col]=st.number_input("Model year (dataset often uses 70–82)",min_value=0,max_value=2100,value=int(default),key=key)
                    else: vals[col]=st.number_input(label,value=default,format="%.2f",key=key)
                else:
                    opts=sorted(series.dropna().astype(str).unique().tolist())
                    vals[col]=st.text_input("Car name (optional)",value="custom vehicle",key=key) if col=="car_name" else st.selectbox(label,opts or ["Unknown"],key=key)
        submitted=st.form_submit_button("🚀 Predict MPG",type="primary",use_container_width=True)
    if submitted:
        try:
            pred=float(pipe.predict(pd.DataFrame([{c:vals[c] for c in features}]))[0])
            st.session_state["last_prediction"]=pred; st.session_state["prediction_done"]=True
        except Exception as e: st.error(f"Prediction failed: {e}")
    if st.session_state.get("prediction_done") and "last_prediction" in st.session_state:
        pred=st.session_state["last_prediction"]
        st.markdown(f"""<div style="border:1px solid {accent};background:linear-gradient(120deg,{panel},{panel2});border-radius:22px;padding:24px;margin-top:16px;animation:rise .55s ease-out both"><div style="color:{accent2};font-weight:800;letter-spacing:2px;font-size:.75rem">PREDICTION COMPLETE</div><div style="font-size:1.05rem;color:{muted};margin-top:7px">Estimated fuel efficiency</div><div style="font-size:3.2rem;font-weight:900;color:{fg}">{pred:.2f} <span style="font-size:1.15rem">MPG</span></div><div style="font-size:1rem;color:{muted}">Approximately {pred*.425144:.2f} km/L</div></div>""",unsafe_allow_html=True)
        st.caption("Predictions are estimates. Actual mileage varies with vehicle condition and driving conditions.")
else:
    st.markdown(f'<div style="background:{panel};border:1px dashed {border};border-radius:18px;padding:22px;margin-top:8px"><b>Your model dashboard is waiting</b><div style="color:{muted};margin-top:5px">Click Train & compare models to unlock the prediction form.</div></div>',unsafe_allow_html=True)

st.divider()
st.markdown(f'<div style="text-align:center;color:{muted};font-size:.82rem">AUTO MPG · ML REGRESSION STUDIO · Streamlit + scikit-learn</div>',unsafe_allow_html=True)
