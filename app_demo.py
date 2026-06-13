"""
AfriAgri Survey Intelligence System
African Agrifood Intelligence Initiative
Regional Agrifood Research Consortium
Analyst: Survey Research Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re, os, warnings, joblib
from io import BytesIO
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AfriAgri Intelligence System",
    page_icon="🌍", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Palette ────────────────────────────────────────────────────────────────
G = "#1D9E75"; T = "#0F6E56"; GO = "#EF9F27"
CO = "#D85A30"; PU = "#7F77DD"; GR = "#888780"
DK = "#1A1A2E"; RED = "#8B0000"

PCOLORS = {"Critical": CO, "High": GO, "Moderate": G, "Low": GR}
RCOLORS = {"High": CO, "Moderate": GO, "Low": G}
UCOLORS = {"Critical": RED, "High": CO, "Routine": GO, "Low": G}
REGCOL  = {"West Africa": G, "East Africa": T, "Central Africa": GO,
           "Southern Africa": CO, "Northern Africa": PU}
ISO = {"Algeria":"DZA","Botswana":"BWA","Cameroon":"CMR","Chad":"TCD",
       "DR Congo":"COD","Egypt":"EGY","Ethiopia":"ETH","Gabon":"GAB",
       "Ghana":"GHA","Kenya":"KEN","Morocco":"MAR","Nigeria":"NGA",
       "Senegal":"SEN","Sierra Leone":"SLE","South Africa":"ZAF",
       "Tanzania":"TZA","Tunisia":"TUN","Uganda":"UGA","Zambia":"ZMB","Zimbabwe":"ZWE"}

# ── Column constants ───────────────────────────────────────────────────────
BAR  = ["Finance_Barrier_Score","Land_Access_Barrier_Score","Climate_Risk_Score","Technology_Barrier_Score"]
READ = ["Staffing_Readiness_Score","Communication_Readiness_Score","Women_Involvement_Score",
        "Hybrid_Platform_Readiness_Score","Strategic_Alignment_Score"]
NEED = ["Funding_Support_Need","Capacity_Building_Need","Technology_Intervention_Need",
        "Land_Ownership_Importance","Early_Warning_Importance"]
PART = ["Government_Partner","Private_Sector_Partner","International_Org_Partner",
        "Academia_Partner","Farmers_Association_Partner","CSO_Partner"]
SUS  = ["Resource_Mobilization_Score","Funding_Diversification_Score",
        "Partnership_Strategy_Score","Knowledge_Sharing_Score"]
MEL  = ["KPI_Alignment_Score","Impact_Measurement_Score",
        "Gender_Data_Promotion_Score","Continuous_Learning_Score"]
GOV  = ["Framework_Inclusivity_Score","Governance_Transparency_Score","Membership_Clarity_Score",
        "Stakeholder_Participation_Score","Trust_Building_Score"]
TEXT = ["Lessons_Learned_Text","Innovation_Recommendations_Text","Inclusivity_Recommendations_Text"]
TNAMES = ["Lessons Learned","Innovation Recommendations","Inclusivity Recommendations"]
DEMO = ["Gender","Age_Range","Location_Type","African_Region",
        "Organization_Type","Value_Chain","Reporting_Level","Disability_Status"]
CFEATS = BAR + NEED + READ + ["Vulnerability_Index","Readiness_Gap","Disability_Flag","Rural_Female"]

BAR_LBL = ["Finance","Land Access","Climate Risk","Technology"]
SUS_LBL = ["Resource Mobilisation","Funding Diversification","Partnership Strategy","Knowledge Sharing"]
MEL_LBL = ["KPI Alignment","Impact Measurement","Gender Data","Continuous Learning"]
GOV_LBL = ["Framework Inclusivity","Gov. Transparency","Membership Clarity","Stakeholder Participation","Trust Building"]

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.main{{background:#F8F9FA;}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,{DK} 0%,#16213E 100%);}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] div{{color:white!important;}}
.kpi{{background:white;border-radius:12px;padding:16px 12px;border-left:4px solid {G};
    box-shadow:0 2px 8px rgba(0,0,0,.06);text-align:center;height:100%;}}
.kv{{font-size:1.75rem;font-weight:700;line-height:1.1;}}
.kl{{font-size:.72rem;color:#666;font-weight:500;margin-top:3px;
    text-transform:uppercase;letter-spacing:.04em;}}
.ks{{font-size:.71rem;margin-top:3px;color:#999;}}
.sh{{background:linear-gradient(135deg,{T},{G});color:white;padding:10px 17px;
    border-radius:10px;margin-bottom:14px;font-weight:600;font-size:.97rem;}}
.ctx{{font-size:.8rem;color:#666;font-style:italic;margin:-8px 0 10px;}}
.ok{{background:#EBF9F4;border-left:4px solid {G};padding:9px 13px;
    border-radius:6px;font-size:.84rem;color:#111;margin:5px 0;}}
.wn{{background:#FFF8EC;border-left:4px solid {GO};padding:9px 13px;
    border-radius:6px;font-size:.84rem;color:#111;margin:5px 0;}}
.cr{{background:#FEF0EC;border-left:4px solid {CO};padding:9px 13px;
    border-radius:6px;font-size:.84rem;color:#111;margin:5px 0;}}
.ib{{background:#EEF3FF;border-left:4px solid {PU};padding:9px 13px;
    border-radius:6px;font-size:.84rem;color:#111;margin:5px 0;}}
.rc{{background:white;border:1px solid #E8E8E8;border-radius:10px;
    padding:11px 14px;margin:5px 0;box-shadow:0 1px 4px rgba(0,0,0,.04);}}
.tg{{display:inline-block;padding:2px 8px;border-radius:12px;
    font-size:.68rem;font-weight:600;margin-right:4px;}}
.hdr{{background:linear-gradient(135deg,{DK},{T});color:white;
    padding:15px 21px;border-radius:12px;margin-bottom:16px;}}
</style>""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def kpi(c, v, l, s, bc=G):
    c.markdown(f'<div class="kpi" style="border-left-color:{bc}"><div class="kv" style="color:{bc}">{v}</div><div class="kl">{l}</div><div class="ks">{s}</div></div>', unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def ctx(t): st.markdown(f'<div class="ctx">{t}</div>', unsafe_allow_html=True)
def ok(m):  st.markdown(f'<div class="ok">✅ {m}</div>', unsafe_allow_html=True)
def wn(m):  st.markdown(f'<div class="wn">⚠️ {m}</div>', unsafe_allow_html=True)
def cr(m):  st.markdown(f'<div class="cr">🔴 {m}</div>', unsafe_allow_html=True)
def ib(m):  st.markdown(f'<div class="ib">ℹ️ {m}</div>', unsafe_allow_html=True)

def pf(fig, h=300):
    fig.update_layout(height=h, margin=dict(t=10,b=10,l=0,r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter,sans-serif", size=11))
    fig.update_xaxes(gridcolor="#EEEEEE", linecolor="#DDD")
    fig.update_yaxes(gridcolor="#EEEEEE", linecolor="#DDD")
    return fig

def dl(df_exp, lbl, fname):
    st.download_button(f"⬇️ Download {lbl}", df_exp.to_csv(index=False).encode(),
        fname, "text/csv", use_container_width=True)

def wc_fig(wfreq, title, color):
    if HAS_WC and wfreq:
        wc = WordCloud(width=700, height=320, background_color="white",
            colormap="YlGn", max_words=40, prefer_horizontal=0.9
        ).generate_from_frequencies(dict(wfreq))
        fig, ax = plt.subplots(figsize=(7,3.2))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
        plt.tight_layout(); return fig, "mpl"
    if not wfreq: return None, None
    words, freqs = zip(*wfreq[:30])
    mf = max(freqs); sizes = [10+(f/mf)*38 for f in freqs]
    np.random.seed(42)
    x = np.random.uniform(0.04,0.96,len(words))
    y = np.random.uniform(0.08,0.92,len(words))
    fig = go.Figure(go.Scatter(x=x,y=y,mode="text",text=list(words),
        textfont=dict(size=sizes,color=color),
        hovertext=[f"{w}: {f}" for w,f in zip(words,freqs)],hoverinfo="text"))
    fig.update_layout(height=280,title=dict(text=title,font=dict(size=11)),
        xaxis=dict(showgrid=False,showticklabels=False,zeroline=False,range=[0,1]),
        yaxis=dict(showgrid=False,showticklabels=False,zeroline=False,range=[0,1]),
        margin=dict(t=35,b=5,l=5,r=5),plot_bgcolor="white",paper_bgcolor="white")
    return fig, "plotly"


# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading AfriAgri dataset…")
def load_data():
    for p in ["afriagri_enriched_v2.csv","afriagri_clean.csv"]:
        if os.path.exists(p): df = pd.read_csv(p); break
    else:
        df = pd.read_excel("AfriAgri_Survey_Dataset.xlsx")
        df = df.drop(columns=["Full_Name","Phone_Number","Email_Address"],errors="ignore")

    df["Country_ISO"] = df["Country"].map(ISO)

    def norm(s,h=True): n=(s-s.min())/(s.max()-s.min())*100; return n if h else 100-n

    # ── Base derived columns (must exist before WAEI / Risk blocks) ───────
    if "Avg_Barrier_Score" not in df.columns:
        df["Avg_Barrier_Score"] = df[BAR].mean(axis=1).round(3)
    if "Avg_Readiness_Score" not in df.columns:
        df["Avg_Readiness_Score"] = df[READ].mean(axis=1).round(3)
    if "Avg_Need_Score" not in df.columns:
        df["Avg_Need_Score"] = df[NEED].mean(axis=1).round(3)
    if "Readiness_Gap" not in df.columns:
        df["Readiness_Gap"] = (df["Avg_Barrier_Score"] - df["Avg_Readiness_Score"]).round(3)
    if "Rural_Female" not in df.columns:
        df["Rural_Female"] = ((df["Gender"] == "Female") & (df["Location_Type"] == "Rural")).astype(int)
    if "Disability_Flag" not in df.columns:
        df["Disability_Flag"] = (df["Disability_Status"] == "Yes").astype(int)
    if "Total_Partners" not in df.columns:
        df["Total_Partners"] = df[PART].sum(axis=1)
    if "Participation_Score" not in df.columns:
        df["Participation_Score"] = df[["Stakeholder_Participation_Score", "Women_Involvement_Score"]].mean(axis=1)

    # WAEI
    if "WAEI_Score" not in df.columns:
        df["WAEI_Participation"]  = norm(df["Women_Involvement_Score"])
        df["WAEI_Governance"]     = norm(df["Governance_Readiness_Score"])
        df["WAEI_Resilience"]     = norm(df["Avg_Barrier_Score"],False)
        df["WAEI_Sustainability"] = norm(df["Sustainability_Score"])
        df["WAEI_Inclusion"]      = norm(df["Framework_Inclusivity_Score"])
        df["WAEI_Score"] = (df["WAEI_Participation"]*0.30+df["WAEI_Governance"]*0.25+
            df["WAEI_Resilience"]*0.20+df["WAEI_Sustainability"]*0.15+df["WAEI_Inclusion"]*0.10).round(2)
    else:
        for col,src,h in [("WAEI_Participation","Women_Involvement_Score",True),
            ("WAEI_Governance","Governance_Readiness_Score",True),
            ("WAEI_Resilience","Avg_Barrier_Score",False),
            ("WAEI_Sustainability","Sustainability_Score",True),
            ("WAEI_Inclusion","Framework_Inclusivity_Score",True)]:
            if col not in df.columns: df[col]=norm(df[src],h)

    # Risk
    if "Exclusion_Risk_Score" not in df.columns:
        df["Exclusion_Risk_Score"] = (df["Avg_Barrier_Score"]*0.35+(5-df["Avg_Readiness_Score"])*0.25+
            df["Vulnerability_Index"]*0.25+df["Disability_Flag"]*0.50+df["Rural_Female"]*0.30).round(3)
    if "Risk_Level" not in df.columns:
        rq = df["Exclusion_Risk_Score"].quantile([0.33,0.66])
        df["Risk_Level"] = pd.cut(df["Exclusion_Risk_Score"],
            bins=[-np.inf,rq.iloc[0],rq.iloc[1],np.inf],labels=["Low","Moderate","High"])

    # Prescriptions
    if "Urgency" not in df.columns:
        def px_fn(row):
            recs=[]; u="Routine"
            if row["Finance_Barrier_Score"]>=4: recs.append("Agricultural credit & microfinance access"); u="High"
            if row["Technology_Barrier_Score"]>=4: recs.append("Digital literacy & agri-tech platform access"); u="High"
            if row["Climate_Risk_Score"]>=4: recs.append("Climate-smart agriculture & early warning systems")
            if row["Land_Access_Barrier_Score"]>=4: recs.append("Land rights advocacy & documentation support")
            if row["Governance_Readiness_Score"]<3.0: recs.append("Governance capacity building")
            if row["Gender"]=="Female" and row["Location_Type"]=="Rural": recs.append("Rural women extension services")
            if row["Vulnerability_Index"]>=4.0: recs.append("Emergency welfare inclusion support"); u="Critical"
            if row.get("Disability_Status","No")=="Yes": recs.append("Disability-inclusive programme design")
            if row["Capacity_Building_Need"]>=4: recs.append("Capacity building programme enrolment")
            if not recs: recs.append("Monitor — no critical barriers detected"); u="Low"
            return pd.Series({"Recommendations":"|".join(recs),"Urgency":u,"Num_Interventions":len(recs)})
        df = pd.concat([df, df.apply(px_fn,axis=1)], axis=1)

    # Remaining derived columns (safe to run again; all guarded by if-not-in)
    if "Num_Interventions" not in df.columns:
        df["Num_Interventions"] = 0
    if "Priority_Ordinal" not in df.columns:
        df["Priority_Ordinal"] = df["Intervention_Priority"].map({"Low":0,"Moderate":1,"High":2,"Critical":3})

    # Sub-scores
    df["MEL_Score"]  = df[MEL].mean(axis=1).round(3)
    df["Sus_Score"]  = df[SUS].mean(axis=1).round(3)
    df["Gov_Score"]  = df[GOV].mean(axis=1).round(3)

    # Clustering
    if "Cluster" not in df.columns:
        sc = StandardScaler(); Xc = sc.fit_transform(df[CFEATS])
        km = KMeans(n_clusters=2,random_state=42,n_init=10)
        df["Cluster"] = km.fit_predict(Xc)
        pc = PCA(n_components=2,random_state=42); Xp = pc.fit_transform(Xc)
        df["PCA1"]=Xp[:,0]; df["PCA2"]=Xp[:,1]
    elif "PCA1" not in df.columns:
        sc = StandardScaler(); Xc = sc.fit_transform(df[CFEATS])
        pc = PCA(n_components=2,random_state=42); Xp = pc.fit_transform(Xc)
        df["PCA1"]=Xp[:,0]; df["PCA2"]=Xp[:,1]

    if "Cluster_Name" not in df.columns:
        hi = df.groupby("Cluster")["Vulnerability_Index"].mean().idxmax()
        df["Cluster_Name"] = df["Cluster"].map({hi:"High-Vulnerability Group",1-hi:"Lower-Risk Engaged Group"})

    # Word frequencies
    STOP = {"and","for","the","are","that","with","this","from","not","has","have","been",
            "will","our","its","their","more","also","can","into","which","should","about",
            "these","other","such","they","them","through"}
    allw = []; cw = {c:[] for c in TEXT}
    for col in TEXT:
        for txt in df[col].dropna():
            ws = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b",txt.lower()) if w not in STOP]
            allw.extend(ws); cw[col].extend(ws)
    df.attrs["wf"]  = Counter(allw).most_common(50)
    df.attrs["cwf"] = {c:Counter(v).most_common(30) for c,v in cw.items()}
    return df


@st.cache_resource
def load_models():
    mf = {"cal":"models/priority_calibrated_proba.pkl",
          "vuln":"models/vulnerability_regressor.pkl",
          "le":"models/label_encoder.pkl"}
    return {k:joblib.load(v) for k,v in mf.items() if os.path.exists(v)}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌍 AfriAgri Intelligence")
    st.markdown("**Regional Agrifood Research Consortium**  \n*Research & Analytics Division*")
    st.divider()
    PAGES = ["01 · Executive Overview","02 · Regional Intelligence",
             "03 · Barrier & Risk Analytics","04 · Predictive Analytics",
             "05 · Stakeholder Segments","06 · Intervention Recommender",
             "07 · Text Intelligence","08 · MEL Framework",
             "09 · Sustainability & Partnerships"]
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Global Filters**")
    df_raw = load_data()
    sel_r = st.multiselect("Region", sorted(df_raw["African_Region"].unique()),
                           default=sorted(df_raw["African_Region"].unique()))
    sel_g = st.multiselect("Gender", sorted(df_raw["Gender"].unique()),
                           default=sorted(df_raw["Gender"].unique()))
    sel_o = st.multiselect("Organisation", sorted(df_raw["Organization_Type"].unique()),
                           default=sorted(df_raw["Organization_Type"].unique()))
    sel_l = st.multiselect("Location", sorted(df_raw["Location_Type"].unique()),
                           default=sorted(df_raw["Location_Type"].unique()))
    st.divider()
    st.caption(f"**{len(df_raw):,}** respondents · **{df_raw['Country'].nunique()}** countries")
    st.caption("AfriAgri Intelligence System v2.0")

df = df_raw.copy()
if sel_r: df = df[df["African_Region"].isin(sel_r)]
if sel_g: df = df[df["Gender"].isin(sel_g)]
if sel_o: df = df[df["Organization_Type"].isin(sel_o)]
if sel_l: df = df[df["Location_Type"].isin(sel_l)]
mdl = load_models()
if len(df)==0: st.warning("No data matches filters."); st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# P01 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "01 · Executive Overview":
    st.markdown(f"""<div class="hdr"><h2 style="margin:0;font-size:1.3rem;font-weight:700">
    AfriAgri Survey Intelligence System</h2><p style="margin:4px 0 0;opacity:.88;font-size:.87rem">
    African Agrifood Intelligence Initiative · African Union Commission · Research Analytics Department
    </p></div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:14px;line-height:1.6">Continental survey intelligence dashboard. Use the sidebar to filter by region, gender, or organisation type. All pages update automatically.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    crit_n  = int((df["Intervention_Priority"]=="Critical").sum())
    hiR_n   = int((df["Risk_Level"]=="High").sum()) if "Risk_Level" in df.columns else 0
    kpi(c1,f"{len(df):,}","Total Respondents",f"{df['Country'].nunique()} countries · {df['African_Region'].nunique()} regions",T)
    kpi(c2,f"{crit_n:,}","Critical Priority",f"{crit_n/len(df)*100:.1f}% need urgent action",CO)
    kpi(c3,f"{df['Vulnerability_Index'].mean():.2f}/5","Mean Vulnerability","Higher = more exposed to barriers",GO)
    kpi(c4,f"{df['WAEI_Score'].mean():.1f}/100","WAEI Score","Women Agrifood Empowerment Index",G)
    kpi(c5,f"{int(df['Disability_Flag'].sum()):,}","With Disability",f"{df['Disability_Flag'].mean()*100:.1f}% need inclusive design",PU)
    st.markdown("<br>",unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1,f"{hiR_n:,}","High Exclusion Risk",f"{hiR_n/len(df)*100:.1f}% face critical exclusion",CO)
    kpi(c2,f"{df['Participation_Score'].mean()/5*100:.0f}%","Participation Readiness","Estimated programme engagement",G)
    kpi(c3,f"{df['Governance_Readiness_Score'].mean():.2f}/5","Governance Readiness","Framework capacity",T)
    kpi(c4,f"{df['MEL_Score'].mean():.2f}/5","MEL Readiness","Monitoring & evaluation capacity",PU)
    kpi(c5,f"{df['Num_Interventions'].mean():.1f}","Avg Interventions","Needed per respondent",GO)
    st.markdown("<br>",unsafe_allow_html=True)

    cl,cm,cr2 = st.columns([2,2,1.5])
    with cl:
        sec("📊 Intervention Priority Breakdown")
        ctx("Number of respondents in each urgency level.")
        p = df["Intervention_Priority"].value_counts().reindex(["Critical","High","Moderate","Low"],fill_value=0).reset_index()
        p.columns=["Priority","Count"]; p["Pct"]=(p["Count"]/len(df)*100).round(1)
        fig=px.bar(p,x="Priority",y="Count",color="Priority",
            text=p["Pct"].map(lambda x:f"{x:.1f}%"),color_discrete_map=PCOLORS)
        fig.update_traces(textposition="outside")
        fig=pf(fig,280); fig.update_layout(showlegend=False,xaxis_title="",
            yaxis=dict(gridcolor="#EEEEEE",title="Respondents"))
        st.plotly_chart(fig,use_container_width=True)

    with cm:
        sec("🌍 WAEI by African Region")
        ctx("Empowerment score 0–100. Below 50 = structural gaps.")
        wr=df.groupby("African_Region")["WAEI_Score"].mean().sort_values().reset_index()
        fig2=go.Figure(go.Bar(x=wr["WAEI_Score"],y=wr["African_Region"],orientation="h",
            text=wr["WAEI_Score"].map(lambda x:f"{x:.1f}"),textposition="outside",
            marker_color=[G if v>=65 else GO if v>=50 else CO for v in wr["WAEI_Score"]]))
        fig2.add_vline(x=50,line_dash="dash",line_color=GR,line_width=1.2,
            annotation_text="Target",annotation_position="top right")
        fig2=pf(fig2,280); fig2.update_layout(xaxis=dict(range=[0,100],gridcolor="#EEEEEE",title="WAEI Score"),yaxis_title="")
        st.plotly_chart(fig2,use_container_width=True)

    with cr2:
        sec("⚡ Key Alerts")
        tv=df.groupby("African_Region")["Vulnerability_Index"].mean().idxmax()
        lw=df.groupby("African_Region")["WAEI_Score"].mean().idxmin()
        hb=(df["Avg_Barrier_Score"]>=4).mean()*100 if "Avg_Barrier_Score" in df.columns else 0
        uc=int((df["Urgency"]=="Critical").sum()) if "Urgency" in df.columns else 0
        cr(f"<strong>{tv}</strong> — highest vulnerability. Priority for programme resources.")
        cr(f"<strong>{uc:,}</strong> respondents flagged Critical urgency.")
        wn(f"<strong>{lw}</strong> — lowest empowerment score.")
        ok(f"<strong>{hb:.1f}%</strong> face high barrier burden (≥4/5).")

    st.markdown("<br>",unsafe_allow_html=True)
    cl2,cr3 = st.columns(2)
    with cl2:
        sec("📋 Urgency by Region")
        ctx("Urgency is based on the combination and severity of each person's barriers.")
        if "Urgency" in df.columns:
            ug=df.groupby(["African_Region","Urgency"]).size().reset_index(name="Count")
            fig3=px.bar(ug,x="African_Region",y="Count",color="Urgency",
                color_discrete_map=UCOLORS,barmode="stack")
            fig3=pf(fig3,270); fig3.update_layout(xaxis_title="",yaxis_title="Respondents",
                xaxis_tickangle=-20,legend_title="Urgency")
            st.plotly_chart(fig3,use_container_width=True)
    with cr3:
        sec("👥 Respondents by Organisation & Gender")
        ctx("Who participated — by organisation type and gender.")
        dm=df.groupby(["Organization_Type","Gender"]).size().reset_index(name="Count")
        fig4=px.bar(dm,x="Organization_Type",y="Count",color="Gender",
            color_discrete_map={"Female":G,"Male":PU},barmode="stack")
        fig4=pf(fig4,270); fig4.update_layout(xaxis_title="",yaxis_title="Respondents",xaxis_tickangle=-25)
        st.plotly_chart(fig4,use_container_width=True)

    st.divider()
    dl(df[["Respondent_ID","African_Region","Country","Gender","Age_Range","Organization_Type",
        "Intervention_Priority","Vulnerability_Index","WAEI_Score","Risk_Level","Urgency",
        "Num_Interventions","Recommendations"]],
        "Executive Summary","afriagri_executive_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P02 — REGIONAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "02 · Regional Intelligence":
    sec("🌍 Regional Intelligence — Geographic Performance Breakdown")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">Understand how AfriAgri priorities, vulnerability, and empowerment vary across Africa\'s five regions and 20 countries. Use this to guide where resources should be deployed first.</div>',unsafe_allow_html=True)

    cl,cr2=st.columns([3,2])
    with cl:
        sec("Vulnerability Index by Country — Africa Map")
        ctx("Darker red = higher vulnerability. These countries need the most urgent attention.")
        cv=df.groupby(["Country","Country_ISO"])["Vulnerability_Index"].mean().reset_index()
        cv.columns=["Country","iso_alpha","Vulnerability"]
        fm=px.choropleth(cv,locations="iso_alpha",color="Vulnerability",hover_name="Country",
            color_continuous_scale=[[0,"#C8F0DC"],[0.5,GO],[1,CO]],
            scope="africa",height=420,labels={"Vulnerability":"Vulnerability\nIndex"})
        fm.update_geos(showframe=False,showcoastlines=True,coastlinecolor="white",
            showland=True,landcolor="#F0F0F0",showocean=True,oceancolor="#E8F4FD")
        fm.update_layout(margin=dict(t=5,b=5,l=0,r=0),
            coloraxis_colorbar=dict(title="Vuln.",len=0.7),paper_bgcolor="white")
        st.plotly_chart(fm,use_container_width=True)

    with cr2:
        sec("Regional Scorecard")
        ctx("At-a-glance summary. Sort by any column.")
        rs=df.groupby("African_Region").agg(
            N=("Respondent_ID","count"),
            Vulnerability=("Vulnerability_Index","mean"),
            WAEI=("WAEI_Score","mean"),
            Critical_Pct=("Intervention_Priority",lambda x:(x=="Critical").mean()*100),
            Avg_Barrier=("Avg_Barrier_Score","mean"),
        ).round(2).reset_index().sort_values("Vulnerability",ascending=False)
        rs.columns=["Region","N","Vulnerability","WAEI","Critical %","Avg Barrier"]
        st.dataframe(rs,use_container_width=True,height=340,
            column_config={"WAEI":st.column_config.ProgressColumn("WAEI",min_value=0,max_value=100,format="%.1f"),
                "Vulnerability":st.column_config.NumberColumn(format="%.2f"),
                "Critical %":st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Barrier":st.column_config.NumberColumn(format="%.2f")})

    c1,c2=st.columns(2)
    with c1:
        sec("Governance Readiness vs Vulnerability")
        ctx("Bottom-right = strong governance + low vulnerability. Best performers.")
        sc=df.groupby("African_Region").agg(
            Vulnerability=("Vulnerability_Index","mean"),
            Governance=("Governance_Readiness_Score","mean"),
            Count=("Respondent_ID","count")).reset_index()
        fig=px.scatter(sc,x="Governance",y="Vulnerability",size="Count",
            color="African_Region",text="African_Region",color_discrete_map=REGCOL,
            labels={"Governance":"Governance Readiness (1–5)","Vulnerability":"Vulnerability (1–5)"})
        fig.update_traces(textposition="top center",textfont_size=9)
        fig=pf(fig,290); fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("Priority Heatmap by Country")
        ctx("% of respondents in each priority level, per country.")
        cp=df.groupby(["Country","Intervention_Priority"]).size().unstack(fill_value=0)
        cpc=cp.div(cp.sum(axis=1),axis=0)*100
        cols=[c for c in ["Critical","High","Moderate","Low"] if c in cpc.columns]
        fig=px.imshow(cpc[cols],text_auto=".0f",
            color_continuous_scale=[[0,"#F7FBFF"],[0.5,GO],[1,CO]],
            aspect="auto",height=290,zmin=0,zmax=60,labels={"color":"%"})
        fig=pf(fig,290); fig.update_layout(coloraxis_colorbar=dict(title="%",len=0.7))
        st.plotly_chart(fig,use_container_width=True)

    sec("WAEI Component Breakdown by Region")
    ctx("Which dimension of women's empowerment is weakest in each region?")
    wc_cols=["WAEI_Participation","WAEI_Governance","WAEI_Resilience","WAEI_Sustainability","WAEI_Inclusion"]
    wc_lbl=["Participation","Governance","Resilience","Sustainability","Inclusion"]
    if all(c in df.columns for c in wc_cols):
        rc=df.groupby("African_Region")[wc_cols].mean().reset_index()
        rm=rc.melt(id_vars="African_Region",var_name="Component",value_name="Score")
        rm["Component"]=rm["Component"].map(dict(zip(wc_cols,wc_lbl)))
        fig=px.bar(rm,x="African_Region",y="Score",color="Component",barmode="group",
            color_discrete_sequence=[G,T,GO,CO,PU])
        fig=pf(fig,270); fig.update_layout(xaxis_title="",
            yaxis_title="WAEI Component Score (0–100)",xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    st.divider()
    re_exp=df.groupby(["African_Region","Country"]).agg(N=("Respondent_ID","count"),
        Vulnerability=("Vulnerability_Index","mean"),WAEI=("WAEI_Score","mean"),
        Barrier=("Avg_Barrier_Score","mean"),
        Critical_Pct=("Intervention_Priority",lambda x:(x=="Critical").mean()*100)
    ).round(2).reset_index()
    dl(re_exp,"Regional Intelligence","afriagri_regional.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P03 — BARRIER & RISK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "03 · Barrier & Risk Analytics":
    sec("🚧 Barrier & Risk Analytics")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">Scores are on a 1–5 scale where 5 is most severe. Scores above 3.5 require programme intervention. Scores above 4.0 are critical.</div>',unsafe_allow_html=True)

    bm=df[BAR].mean()
    c1,c2,c3,c4=st.columns(4)
    for col,lbl,score,bc in zip([c1,c2,c3,c4],BAR_LBL,bm.values,[CO,GO,G,PU]):
        grade="CRITICAL" if score>=4.0 else "HIGH" if score>=3.5 else "MODERATE" if score>=3.0 else "LOW"
        kpi(col,f"{score:.2f}/5",f"{lbl} Barrier",f"Severity: {grade}",bc)
    st.markdown("<br>",unsafe_allow_html=True)

    cl,cr2=st.columns(2)
    with cl:
        sec("Barrier Severity by Region")
        ctx("Which regions face the most severe barriers? Bars crossing the dashed line need action.")
        br=df.groupby("African_Region")[BAR].mean().reset_index()
        bm2=br.melt(id_vars="African_Region",var_name="Barrier",value_name="Score")
        bm2["Barrier"]=bm2["Barrier"].str.replace("_Barrier_Score","").str.replace("_Risk_Score","").str.replace("_"," ")
        fig=px.bar(bm2,x="African_Region",y="Score",color="Barrier",barmode="group",
            color_discrete_sequence=[CO,GO,G,PU])
        fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2,
            annotation_text="Action threshold",annotation_position="top right")
        fig=pf(fig,300); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    with cr2:
        sec("Barriers — Female vs Male Respondents")
        ctx("Does gender affect barrier severity? This shapes programme design priorities.")
        gd=df.groupby("Gender")[BAR].mean().reset_index()
        gm=gd.melt(id_vars="Gender",var_name="Barrier",value_name="Score")
        gm["Barrier"]=gm["Barrier"].str.replace("_Barrier_Score","").str.replace("_Risk_Score","").str.replace("_"," ")
        fig=px.bar(gm,x="Barrier",y="Score",color="Gender",barmode="group",
            color_discrete_map={"Female":G,"Male":PU})
        fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2)
        fig=pf(fig,300); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"))
        st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        sec("Exclusion Risk by Region, Age & Disability")
        ctx("Risk combines barriers, readiness gaps, vulnerability, disability, and rural-female status.")
        if "Risk_Level" in df.columns:
            tab_r,tab_a,tab_d=st.tabs(["By Region","By Age Group","By Disability"])
            for tab,grp in zip([tab_r,tab_a,tab_d],["African_Region","Age_Range","Disability_Status"]):
                with tab:
                    rc=df.groupby([grp,"Risk_Level"]).size().reset_index(name="Count")
                    rc["Risk_Level"]=pd.Categorical(rc["Risk_Level"],categories=["Low","Moderate","High"],ordered=True)
                    fig=px.bar(rc,x=grp,y="Count",color="Risk_Level",
                        color_discrete_map=RCOLORS,barmode="stack")
                    fig=pf(fig,230); fig.update_layout(xaxis_title="",yaxis_title="Respondents",
                        xaxis_tickangle=-20,legend_title="Risk Level")
                    tab.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("Full Barrier Heatmap — All Regions")
        ctx("One view for every region's full barrier profile. Darker = more severe.")
        hm=df.groupby("African_Region")[BAR].mean()
        hm.columns=BAR_LBL
        fig=px.imshow(hm,text_auto=".2f",
            color_continuous_scale=[[0,"#E8F8F1"],[0.5,GO],[1,CO]],
            aspect="auto",zmin=1,zmax=5,labels={"color":"Score"})
        fig=pf(fig,240); fig.update_layout(coloraxis_colorbar=dict(title="Score",len=0.8))
        st.plotly_chart(fig,use_container_width=True)

    sec("Barrier Severity by Value Chain Segment")
    ctx("Different parts of the agrifood chain face different barriers. This shapes sector-specific programmes.")
    vc=df.groupby("Value_Chain")[BAR].mean().reset_index()
    vm=vc.melt(id_vars="Value_Chain",var_name="Barrier",value_name="Score")
    vm["Barrier"]=vm["Barrier"].str.replace("_Barrier_Score","").str.replace("_Risk_Score","").str.replace("_"," ")
    fig=px.bar(vm,x="Value_Chain",y="Score",color="Barrier",barmode="group",
        color_discrete_sequence=[CO,GO,G,PU])
    fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2)
    fig=pf(fig,270); fig.update_layout(xaxis_title="",
        yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-15)
    st.plotly_chart(fig,use_container_width=True)

    st.divider()
    be=df.groupby(["African_Region","Gender","Location_Type"])[BAR+["Avg_Barrier_Score"]].mean().round(2).reset_index()
    dl(be,"Barrier & Risk Data","afriagri_barriers.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P04 — PREDICTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "04 · Predictive Analytics":
    sec("🔮 Predictive Analytics — What the Data Forecasts")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">These insights go beyond describing what happened — they forecast what is likely to happen and which interventions will have the most impact. All predictions are from validated machine learning models.</div>',unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        sec("Predicted Participation Readiness by Region")
        ctx("How likely is each region to actively engage with programmes?")
        pr=(df.groupby("African_Region")["Participation_Score"].mean()/5*100).round(1).sort_values(ascending=False).reset_index()
        pr.columns=["Region","Pct"]
        fig=go.Figure(go.Bar(x=pr["Pct"],y=pr["Region"],orientation="h",
            text=pr["Pct"].map(lambda x:f"{x:.1f}%"),textposition="outside",
            marker_color=[G if v>=70 else GO if v>=55 else CO for v in pr["Pct"]]))
        fig.add_vline(x=60,line_dash="dash",line_color=GR,line_width=1.2,
            annotation_text="Target (60%)",annotation_position="top right")
        fig=pf(fig,270); fig.update_layout(
            xaxis=dict(range=[0,105],gridcolor="#EEEEEE",title="Participation readiness (%)"),yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("Governance Failure Risk by Region")
        ctx("% of respondents with low governance readiness. High % = implementation risk for AfriAgri.")
        gf=df.groupby("African_Region").apply(
            lambda x:(x["Governance_Readiness_Score"]<3.0).mean()*100
        ).sort_values(ascending=False).reset_index()
        gf.columns=["Region","Low_Gov_Pct"]
        fig=go.Figure(go.Bar(x=gf["Low_Gov_Pct"],y=gf["Region"],orientation="h",
            text=gf["Low_Gov_Pct"].map(lambda x:f"{x:.1f}%"),textposition="outside",
            marker_color=[CO if v>=30 else GO if v>=15 else G for v in gf["Low_Gov_Pct"]]))
        fig=pf(fig,270); fig.update_layout(
            xaxis=dict(range=[0,60],gridcolor="#EEEEEE",title="% with low governance readiness"),yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        sec("Funding Demand Forecast by Region")
        ctx("Where is the demand for financial support highest? This guides resource allocation.")
        fn=df.groupby("African_Region")["Funding_Support_Need"].mean().sort_values(ascending=False).reset_index()
        fn.columns=["Region","Need"]
        fig=go.Figure(go.Bar(x=fn["Need"],y=fn["Region"],orientation="h",
            text=fn["Need"].map(lambda x:f"{x:.2f}/5"),textposition="outside",
            marker_color=[CO if v>=4 else GO if v>=3.5 else G for v in fn["Need"]]))
        fig.add_vline(x=3.5,line_dash="dash",line_color=GR,line_width=1.2)
        fig=pf(fig,270); fig.update_layout(
            xaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Mean funding need (1–5)"),yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("Technology Intervention Demand")
        ctx("Where is digital inclusion most urgently needed?")
        tn=df.groupby("African_Region")["Technology_Intervention_Need"].mean().sort_values(ascending=False).reset_index()
        tn.columns=["Region","Need"]
        fig=go.Figure(go.Bar(x=tn["Need"],y=tn["Region"],orientation="h",
            text=tn["Need"].map(lambda x:f"{x:.2f}/5"),textposition="outside",
            marker_color=[CO if v>=4 else GO if v>=3.5 else G for v in tn["Need"]]))
        fig.add_vline(x=3.5,line_dash="dash",line_color=GR,line_width=1.2)
        fig=pf(fig,270); fig.update_layout(
            xaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Mean technology need (1–5)"),yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

    sec("Scenario Simulation — What Happens If AfriAgri Intervenes?")
    ctx("How much does predicted vulnerability drop when specific interventions are delivered? Green = vulnerability reduced. This is the investment case for donors.")
    if "vuln" in mdl:
        m2=mdl["vuln"]
        VN=READ+NEED+PART+["Total_Partners"]
        try:
            bX=df[VN+DEMO].copy(); bp=m2.predict(bX).mean()
            sc_def=[("Staffing_Readiness_Score",+1,"Staffing & capacity readiness improved"),
                    ("Women_Involvement_Score",+1,"Women involvement in governance improved"),
                    ("Communication_Readiness_Score",+1,"Communication systems strengthened"),
                    ("Funding_Support_Need",-1,"Funding support need fully met"),
                    ("Capacity_Building_Need",-1,"Capacity building need fully met")]
            scs={"No intervention (baseline)":bp}
            for col,d,lbl in sc_def:
                if col in bX.columns:
                    Xs=bX.copy(); Xs[col]=np.clip(Xs[col]+d,1,5)
                    scs[lbl]=m2.predict(Xs).mean()
            Xf=bX.copy()
            for c in ["Staffing_Readiness_Score","Women_Involvement_Score","Communication_Readiness_Score"]:
                if c in Xf.columns: Xf[c]=np.clip(Xf[c]+1,1,5)
            scs["Full combined programme intervention"]=m2.predict(Xf).mean()

            sd=pd.DataFrame(list(scs.items()),columns=["Scenario","Pred"])
            sd["Change"]=(sd["Pred"]-bp).round(4)
            sc_col=[GR]+[G if c<-0.001 else CO if c>0.001 else GO for c in sd["Change"][1:]]
            fig=go.Figure(go.Bar(x=sd["Pred"],y=sd["Scenario"],orientation="h",
                text=[f"{v:.3f} ({c:+.3f})" for v,c in zip(sd["Pred"],sd["Change"])],
                textposition="outside",marker_color=sc_col))
            fig.add_vline(x=bp,line_dash="dash",line_color=GR,line_width=1.5,
                annotation_text=f"Baseline: {bp:.3f}",annotation_position="top right")
            fig=pf(fig,330); fig.update_layout(
                xaxis=dict(gridcolor="#EEEEEE",title="Predicted Vulnerability Index"),
                yaxis_title="",margin=dict(r=220))
            st.plotly_chart(fig,use_container_width=True)
            best=sd[sd["Change"]<0].sort_values("Change")
            if len(best)>0:
                b=best.iloc[0]
                ok(f"Most impactful single intervention: <strong>{b['Scenario']}</strong> — reduces vulnerability by {abs(b['Change']):.3f} points.")
        except Exception as e:
            ib(f"Run `AfriAgri_ML_Pipeline_v2.ipynb` and save models to `models/` folder to activate scenario simulation. ({e})")
    else:
        vr=df.groupby("African_Region")["Vulnerability_Index"].mean().sort_values(ascending=False).reset_index()
        vr.columns=["Region","Vulnerability"]
        fig=px.bar(vr,x="Region",y="Vulnerability",color="Vulnerability",
            color_continuous_scale=[[0,G],[0.5,GO],[1,CO]])
        fig=pf(fig,270); st.plotly_chart(fig,use_container_width=True)
        ib("Run `AfriAgri_ML_Pipeline_v2.ipynb` and place output in `models/` folder to activate live scenario simulation.")

    st.divider()
    pe2=df.groupby("African_Region").agg(
        Participation_Pct=("Participation_Score",lambda x:x.mean()/5*100),
        Funding_Need=("Funding_Support_Need","mean"),
        Tech_Need=("Technology_Intervention_Need","mean"),
        Low_Gov_Pct=("Governance_Readiness_Score",lambda x:(x<3.0).mean()*100),
        Avg_Vulnerability=("Vulnerability_Index","mean"),
    ).round(2).reset_index()
    dl(pe2,"Predictive Analytics","afriagri_predictive.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P05 — STAKEHOLDER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "05 · Stakeholder Segments":
    sec("👥 Stakeholder Segments — Who Are the programme's Key Groups?")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">Machine learning identified two natural groups within the 2,500 respondents based on their barrier profiles and vulnerability. The programme needs very different programmes for each group.</div>',unsafe_allow_html=True)

    if "Cluster_Name" not in df.columns or "PCA1" not in df.columns:
        st.warning("Clustering data not available. Load the enriched dataset."); st.stop()

    s0=df[df["Cluster_Name"]=="High-Vulnerability Group"]
    s1=df[df["Cluster_Name"]=="Lower-Risk Engaged Group"]

    cl,cr2=st.columns(2)
    with cl:
        v0=s0["Vulnerability_Index"].mean(); b0=s0["Avg_Barrier_Score"].mean()
        d0=s0["Disability_Flag"].mean()*100; r0=s0["Rural_Female"].mean()*100
        st.markdown(f"""<div style="background:linear-gradient(135deg,#FEF0EC,#FFF);
            border:2px solid {CO};border-radius:14px;padding:20px;margin-bottom:8px">
            <h3 style="color:{CO};margin:0 0 7px">🔴 High-Vulnerability Group</h3>
            <p style="font-size:.84rem;color:#555;margin:0 0 12px">
            <strong>{len(s0):,} respondents ({len(s0)/len(df)*100:.1f}%)</strong> — 
            Face the most severe barriers. Need the most intensive, targeted support programmes.</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{CO}">{v0:.2f}/5</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Vulnerability</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{CO}">{b0:.2f}/5</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Avg Barrier</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{GO}">{d0:.1f}%</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">With Disability</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{GO}">{r0:.1f}%</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Rural Female</div></div>
            </div></div>""",unsafe_allow_html=True)
        cr(f"Needs: financial access, digital inclusion, rural extension, disability accommodation, emergency welfare support.")

    with cr2:
        v1=s1["Vulnerability_Index"].mean(); b1=s1["Avg_Barrier_Score"].mean()
        d1=s1["Disability_Flag"].mean()*100; r1=s1["Rural_Female"].mean()*100
        st.markdown(f"""<div style="background:linear-gradient(135deg,#EBF9F4,#FFF);
            border:2px solid {G};border-radius:14px;padding:20px;margin-bottom:8px">
            <h3 style="color:{G};margin:0 0 7px">🟢 Lower-Risk Engaged Group</h3>
            <p style="font-size:.84rem;color:#555;margin:0 0 12px">
            <strong>{len(s1):,} respondents ({len(s1)/len(df)*100:.1f}%)</strong> — 
            Better positioned but still facing real barriers. Key partners for Governance and knowledge networks.</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{G}">{v1:.2f}/5</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Vulnerability</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{G}">{b1:.2f}/5</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Avg Barrier</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{T}">{d1:.1f}%</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">With Disability</div></div>
            <div style="background:white;border-radius:8px;padding:10px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:{T}">{r1:.1f}%</div>
            <div style="font-size:.7rem;color:#888;text-transform:uppercase">Rural Female</div></div>
            </div></div>""",unsafe_allow_html=True)
        ok(f"Needs: governance training, knowledge-sharing platforms, partnership facilitation, KPI alignment support.")

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sec("Segment Map — How the Two Groups Differ")
        ctx("Each dot is a respondent. Red = high-vulnerability. Green = lower-risk. Separation shows how distinct the groups are.")
        fig=px.scatter(df,x="PCA1",y="PCA2",color="Cluster_Name",
            color_discrete_map={"High-Vulnerability Group":CO,"Lower-Risk Engaged Group":G},
            opacity=0.32,labels={"PCA1":"Dimension 1","PCA2":"Dimension 2","Cluster_Name":"Group"})
        fig.update_traces(marker_size=5)
        fig=pf(fig,310); st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("Side-by-Side Profile Comparison")
        ctx("Where are the biggest differences between the two groups?")
        kc=["Avg_Barrier_Score","Avg_Readiness_Score","Vulnerability_Index",
            "Finance_Barrier_Score","Technology_Barrier_Score","Governance_Readiness_Score",
            "Sustainability_Score","MEL_Score"]
        kl=["Avg Barrier","Avg Readiness","Vulnerability","Finance","Technology",
            "Governance","Sustainability","MEL Score"]
        cp=df.groupby("Cluster_Name")[kc].mean()
        cp.columns=kl
        cm2=cp.reset_index().melt(id_vars="Cluster_Name",var_name="Metric",value_name="Score")
        fig=px.bar(cm2,x="Metric",y="Score",color="Cluster_Name",barmode="group",
            color_discrete_map={"High-Vulnerability Group":CO,"Lower-Risk Engaged Group":G})
        fig=pf(fig,310); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],title="Score (1–5)",gridcolor="#EEEEEE"),
            xaxis_tickangle=-25,legend_title="Segment")
        st.plotly_chart(fig,use_container_width=True)

    sec("Where Are These Groups Located?")
    c1,c2=st.columns(2)
    with c1:
        ctx("Regional concentration of each segment.")
        rc2=df.groupby(["African_Region","Cluster_Name"]).size().reset_index(name="Count")
        fig=px.bar(rc2,x="African_Region",y="Count",color="Cluster_Name",
            color_discrete_map={"High-Vulnerability Group":CO,"Lower-Risk Engaged Group":G},barmode="stack")
        fig=pf(fig,260); fig.update_layout(xaxis_title="",yaxis_title="Respondents",xaxis_tickangle=-20)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        ctx("Organisation type concentration of each segment.")
        oc=df.groupby(["Organization_Type","Cluster_Name"]).size().reset_index(name="Count")
        fig=px.bar(oc,x="Organization_Type",y="Count",color="Cluster_Name",
            color_discrete_map={"High-Vulnerability Group":CO,"Lower-Risk Engaged Group":G},barmode="stack")
        fig=pf(fig,260); fig.update_layout(xaxis_title="",yaxis_title="Respondents",xaxis_tickangle=-25)
        st.plotly_chart(fig,use_container_width=True)

    sec("AfriAgri Priority Actions per Segment")
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"**High-Vulnerability Group — {len(s0):,} respondents**")
        for ico,txt in [("🏦","Priority financial access — microfinance & credit guarantees"),
            ("💻","Digital literacy & mobile platform access in rural areas"),
            ("👩‍🌾","Dedicated rural women extension officer deployment"),
            ("♿","Disability-inclusive registration and programme delivery"),
            ("🆘","Emergency welfare assessment and rapid inclusion support"),
            ("📋","Land rights legal aid and documentation support")]:
            st.markdown(f'<div class="rc">{ico} {txt}</div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Lower-Risk Engaged Group — {len(s1):,} respondents**")
        for ico,txt in [("🤝","Partnership facilitation with government and international bodies"),
            ("📚","Governance training and leadership development"),
            ("🌐","Knowledge-sharing platform access and peer learning"),
            ("📊","KPI alignment and integration into MEL framework"),
            ("🔬","Research and innovation collaboration opportunities"),
            ("📣","advocacy and awareness ambassador programme")]:
            st.markdown(f'<div class="rc">{ico} {txt}</div>',unsafe_allow_html=True)

    st.divider()
    se=df[["Respondent_ID","African_Region","Country","Gender","Age_Range",
        "Location_Type","Organization_Type","Disability_Status","Cluster_Name",
        "Vulnerability_Index","Avg_Barrier_Score","WAEI_Score","Urgency"]].copy()
    dl(se,"Stakeholder Segments","afriagri_segments.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P06 — INTERVENTION RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "06 · Intervention Recommender":
    sec("🤖 Intervention Recommender — Live Profile Assessment")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">Enter any respondent profile to receive an instant priority assessment and tailored intervention plan. Designed for programme officers, field coordinators, and policy teams.</div>',unsafe_allow_html=True)

    with st.form("pf"):
        st.markdown("**Step 1 — Profile**")
        c1,c2,c3=st.columns(3)
        with c1:
            gender=st.selectbox("Gender",["Female","Male"])
            age=st.selectbox("Age range",["18-35","36-45","46-60","61+"])
            loc=st.selectbox("Location",["Rural","Urban","Peri-urban"])
        with c2:
            region=st.selectbox("Region",["West Africa","East Africa","Central Africa","Southern Africa","Northern Africa"])
            org=st.selectbox("Organisation",sorted(df_raw["Organization_Type"].unique()))
            vc=st.selectbox("Value chain",sorted(df_raw["Value_Chain"].unique()))
        with c3:
            dis=st.selectbox("Disability status",["No","Yes"])
            rep=st.selectbox("Reporting level",["National","Regional","Sub-national","Continental"])

        st.divider()
        st.markdown("**Step 2 — Barrier severity** *(1=none, 5=severe)*")
        b1,b2,b3,b4=st.columns(4)
        fin=b1.slider("Finance barrier",1,5,3,help="Access to credit and financial services")
        land=b2.slider("Land access barrier",1,5,3,help="Ability to secure land rights")
        clim=b3.slider("Climate risk",1,5,3,help="Exposure to climate disruptions")
        tech=b4.slider("Technology barrier",1,5,3,help="Access to digital tools and platforms")

        st.markdown("**Step 3 — Readiness & needs** *(1=very low, 5=very high)*")
        r1,r2,r3,r4=st.columns(4)
        staff=r1.slider("Staffing readiness",1,5,3)
        gov_s=r2.slider("Governance readiness",1,5,3)
        cap=r3.slider("Capacity building need",1,5,3)
        vuln=r4.slider("Vulnerability estimate",1,5,3)

        sub=st.form_submit_button("🔍 Generate Assessment",type="primary",use_container_width=True)

    if sub:
        st.markdown("---")
        rc_col,rr_col=st.columns([2,3])
        with rc_col:
            sec("Priority Assessment")
            if "cal" in mdl and "le" in mdl:
                le2=mdl["le"]; cal2=mdl["cal"]
                dr=pd.DataFrame([{"Gender":gender,"Age_Range":age,"Location_Type":loc,
                    "African_Region":region,"Organization_Type":org,"Value_Chain":vc,
                    "Reporting_Level":rep,"Disability_Status":dis}])
                try:
                    pr2=cal2.predict_proba(dr)[0]
                    prd=pd.DataFrame({"Priority":le2.classes_,"Probability":pr2}).sort_values("Probability",ascending=False)
                    for _,row in prd.iterrows():
                        pct2=row["Probability"]*100; pc=PCOLORS.get(row["Priority"],GR)
                        st.markdown(f"""<div style="margin:7px 0">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-weight:600;color:{pc}">{row["Priority"]}</span>
                        <span style="font-weight:600">{pct2:.1f}%</span></div>
                        <div style="background:#F0F0F0;border-radius:4px;height:9px">
                        <div style="background:{pc};width:{pct2}%;height:9px;border-radius:4px"></div>
                        </div></div>""",unsafe_allow_html=True)
                    tp=prd.iloc[0]; tc=PCOLORS.get(tp["Priority"],GR)
                    st.markdown(f'<div class="kpi" style="border-left-color:{tc};margin-top:12px"><div class="kv" style="color:{tc}">{tp["Priority"]}</div><div class="kl">Most Likely Priority</div><div class="ks">{tp["Probability"]*100:.1f}% model confidence</div></div>',unsafe_allow_html=True)
                except Exception as e:
                    ab=(fin+land+clim+tech)/4
                    p2="Critical" if ab>=4 else "High" if ab>=3.5 else "Moderate" if ab>=3 else "Low"
                    pc=PCOLORS[p2]
                    st.markdown(f'<div class="kpi" style="border-left-color:{pc}"><div class="kv" style="color:{pc}">{p2}</div><div class="kl">Estimated Priority</div><div class="ks">Rule-based estimate</div></div>',unsafe_allow_html=True)
            else:
                ab=(fin+land+clim+tech)/4
                p2="Critical" if ab>=4 else "High" if ab>=3.5 else "Moderate" if ab>=3 else "Low"
                pc=PCOLORS[p2]
                st.markdown(f'<div class="kpi" style="border-left-color:{pc}"><div class="kv" style="color:{pc}">{p2}</div><div class="kl">Estimated Priority</div><div class="ks">Based on barrier profile (rule-based)</div></div>',unsafe_allow_html=True)
                ib("Save ML models to `models/` folder to activate calibrated probability estimates.")

            st.markdown("<br>",unsafe_allow_html=True)
            df2=1 if dis=="Yes" else 0; rf2=1 if (gender=="Female" and loc=="Rural") else 0
            ab2=(fin+land+clim+tech)/4; ar2=(staff+gov_s)/2
            rs2=ab2*0.35+(5-ar2)*0.25+vuln*0.25+df2*0.5+rf2*0.3
            rl2="High" if rs2>4.0 else "Moderate" if rs2>3.2 else "Low"
            rc3=CO if rl2=="High" else GO if rl2=="Moderate" else G
            st.markdown(f'<div class="kpi" style="border-left-color:{rc3}"><div class="kv" style="color:{rc3}">{rl2}</div><div class="kl">Exclusion Risk Level</div><div class="ks">Composite risk score: {rs2:.2f}</div></div>',unsafe_allow_html=True)

        with rr_col:
            sec("Tailored Recommendations")
            recs2=[]
            if fin>=4: recs2.append(("🏦","Agricultural credit facilities & microfinance access","Finance","Critical" if fin==5 else "High"))
            if tech>=4: recs2.append(("💻","Digital literacy training & agri-tech platform access","Technology","High"))
            if clim>=4: recs2.append(("🌱","Climate-smart agriculture & early warning systems","Climate","High"))
            if land>=4: recs2.append(("📋","Land rights legal aid & documentation support","Land","High"))
            if gov_s<3: recs2.append(("🏛️","Governance capacity building & leadership training","Governance","High"))
            if gender=="Female" and loc=="Rural": recs2.append(("👩‍🌾","Rural women extension services & cooperative support","Inclusion","High"))
            if vuln>=4: recs2.append(("🆘","Emergency welfare inclusion & rapid support","Welfare","Critical"))
            if dis=="Yes": recs2.append(("♿","Disability-inclusive programme design & accommodation","Disability","High"))
            if cap>=4: recs2.append(("📚","Capacity building programme enrolment","Capacity","Medium"))
            if age=="18-35": recs2.append(("🌟","Youth Agrifood Leadership Programme","Youth","Medium"))
            if not recs2: recs2.append(("✅","Maintain engagement — no critical barriers detected","Monitoring","Low"))
            ub={"Critical":"#FEF0EC","High":"#FFF8EC","Medium":"#F0FFF8","Low":"#F8F8F8"}
            ut={"Critical":CO,"High":GO,"Medium":G,"Low":GR}
            for ico,txt,cat,urg in recs2:
                uc2=ut.get(urg,GR); bg2=ub.get(urg,"#F8F8F8")
                st.markdown(f"""<div class="rc" style="background:{bg2};border-color:{uc2}22">
                <div style="display:flex;gap:10px;align-items:flex-start">
                <span style="font-size:1.2rem">{ico}</span>
                <div><div style="font-weight:500;font-size:.87rem">{txt}</div>
                <div style="margin-top:4px">
                <span class="tg" style="background:{uc2}22;color:{uc2}">{urg}</span>
                <span class="tg" style="background:#F0F0F0;color:#666">{cat}</span>
                </div></div></div></div>""",unsafe_allow_html=True)
            nc=sum(1 for r in recs2 if r[3]=="Critical")
            nh=sum(1 for r in recs2 if r[3]=="High")
            smsg=f"<strong>{len(recs2)} intervention(s) recommended.</strong> "
            if nc>0: smsg+=f"{nc} critical — immediate action required. "
            if nh>0: smsg+=f"{nh} high priority — action within 30 days."
            cls="cr" if nc>0 else "wn" if nh>0 else "ok"
            st.markdown(f'<div class="{cls}" style="margin-top:10px">{smsg}</div>',unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# P07 — TEXT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "07 · Text Intelligence":
    sec("💬 Text Intelligence — Themes from Open Responses")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">2,500 respondents shared lessons learned, innovation ideas, and inclusivity recommendations. This section shows what the continent\'s agrifood community believes must prioritise — in their own words.</div>',unsafe_allow_html=True)

    wf=df_raw.attrs.get("wf",[]); cwf=df_raw.attrs.get("cwf",{})

    sec("Most Frequently Mentioned Words — All Responses Combined")
    ctx("Larger = mentioned more often. These are the priorities the survey community cares about most.")
    wfig,wtp=wc_fig(wf,"AfriAgri Survey — All Open Responses",T)
    if wfig is not None:
        if wtp=="mpl": st.pyplot(wfig,use_container_width=True)
        else: st.plotly_chart(wfig,use_container_width=True)

    sec("Word Clouds by Response Category")
    t1,t2,t3=st.tabs(TNAMES)
    for tab,col,nm,cl2 in zip([t1,t2,t3],TEXT,TNAMES,[G,GO,PU]):
        with tab:
            wf2=cwf.get(col,[])
            wf3,wt2=wc_fig(wf2,nm,cl2)
            if wf3 is not None:
                if wt2=="mpl": tab.pyplot(wf3,use_container_width=True)
                else: tab.plotly_chart(wf3,use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec("Theme Distribution — Region, Gender & Organisation")
    for col,nm in zip(TEXT,TNAMES):
        if col not in df.columns: continue
        st.markdown(f"**{nm}**")
        tr,tg,to=st.tabs(["By Region","By Gender","By Organisation"])
        with tr:
            cx=pd.crosstab(df["African_Region"],df[col],normalize="index").mul(100).round(1)
            cx.columns=[" ".join(c.split()[:6])+"…" if len(c.split())>6 else c for c in cx.columns]
            fig=px.imshow(cx,text_auto=".1f",
                color_continuous_scale=[[0,"#F7FBFF"],[0.5,G],[1,T]],
                aspect="auto",height=200,labels={"color":"%"})
            fig=pf(fig,200); fig.update_layout(xaxis_tickangle=-25,
                coloraxis_colorbar=dict(title="%",len=0.7))
            tr.plotly_chart(fig,use_container_width=True)
        with tg:
            cg=pd.crosstab(df["Gender"],df[col],normalize="index").mul(100).round(1)
            cg.columns=[" ".join(c.split()[:5])+"…" if len(c.split())>5 else c for c in cg.columns]
            gm2=cg.reset_index().melt(id_vars="Gender",var_name="Theme",value_name="Pct")
            fig=px.bar(gm2,x="Theme",y="Pct",color="Gender",
                color_discrete_map={"Female":G,"Male":PU},barmode="group",height=230)
            fig=pf(fig,230); fig.update_layout(xaxis_title="",yaxis_title="%",xaxis_tickangle=-20)
            tg.plotly_chart(fig,use_container_width=True)
        with to:
            co2=pd.crosstab(df["Organization_Type"],df[col],normalize="index").mul(100).round(1)
            co2.columns=[" ".join(c.split()[:5])+"…" if len(c.split())>5 else c for c in co2.columns]
            fig=px.imshow(co2,text_auto=".1f",
                color_continuous_scale=[[0,"#FFFBF0"],[0.5,GO],[1,CO]],
                aspect="auto",height=230,labels={"color":"%"})
            fig=pf(fig,230); fig.update_layout(xaxis_tickangle=-25,
                coloraxis_colorbar=dict(title="%",len=0.7))
            to.plotly_chart(fig,use_container_width=True)

    st.divider()
    te=df.groupby("African_Region")[TEXT].agg(lambda x:x.value_counts().index[0]).reset_index()
    te.columns=["Region"]+[n+" (dominant theme)" for n in TNAMES]
    dl(te,"Text Intelligence Summary","afriagri_text.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P08 — MEL FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "08 · MEL Framework":
    sec("📈 MEL Framework — Monitoring, Evaluation & Learning")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">The MEL framework is one of AfriAgri\'s core deliverables. This page gives a clear picture of MEL readiness, KPI alignment capacity, and governance quality across all regions — the intelligence needed to design a MEL system that actually works on the ground.</div>',unsafe_allow_html=True)

    mm=df[MEL].mean()
    c1,c2,c3,c4=st.columns(4)
    for col,lbl,score,bc in zip([c1,c2,c3,c4],MEL_LBL,mm.values,[G,T,PU,GO]):
        grade="Strong" if score>=4.0 else "Moderate" if score>=3.5 else "Needs attention"
        kpi(col,f"{score:.2f}/5",lbl,f"MEL capacity: {grade}",bc)

    st.markdown("<br>",unsafe_allow_html=True)
    om=df["MEL_Score"].mean() if "MEL_Score" in df.columns else mm.mean()
    if om>=4.0: ok(f"Overall MEL readiness is strong ({om:.2f}/5). System is ready for implementation with standard training.")
    elif om>=3.5: wn(f"Overall MEL readiness is moderate ({om:.2f}/5). Capacity building on data collection and reporting needed before full implementation.")
    else: cr(f"Overall MEL readiness is low ({om:.2f}/5). Significant MEL capacity building required before programme launch.")

    c1,c2=st.columns(2)
    with c1:
        sec("MEL Readiness by Region")
        ctx("Which regions are most ready to implement monitoring and evaluation systems?")
        mr=df.groupby("African_Region")[MEL].mean().reset_index()
        mm2=mr.melt(id_vars="African_Region",var_name="Metric",value_name="Score")
        mm2["Metric"]=mm2["Metric"].map(dict(zip(MEL,MEL_LBL)))
        fig=px.bar(mm2,x="African_Region",y="Score",color="Metric",barmode="group",
            color_discrete_sequence=[G,T,PU,GO])
        fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2,
            annotation_text="Target",annotation_position="top right")
        fig=pf(fig,290); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        sec("MELAR Readiness Score Distribution")
        ctx("How spread out is MELAR readiness across all respondents? A bell curve is healthy.")
        fig=px.histogram(df,x="MELAR_Readiness_Score",nbins=20,color_discrete_sequence=[T])
        fig.add_vline(x=df["MELAR_Readiness_Score"].mean(),line_dash="dash",line_color=CO,line_width=1.5,
            annotation_text=f"Mean: {df['MELAR_Readiness_Score'].mean():.2f}",annotation_position="top right")
        fig=pf(fig,290); fig.update_layout(showlegend=False,
            xaxis_title="MELAR Readiness Score",yaxis=dict(gridcolor="#EEEEEE",title="Respondents"))
        st.plotly_chart(fig,use_container_width=True)

    sec("Governance Components — Where Are the Gaps?")
    ctx("Governance depends on all five components. This shows where the weaknesses are by region.")
    c1,c2=st.columns(2)
    with c1:
        gr=df.groupby("African_Region")[GOV].mean().reset_index()
        gm2=gr.melt(id_vars="African_Region",var_name="Component",value_name="Score")
        gm2["Component"]=gm2["Component"].map(dict(zip(GOV,GOV_LBL)))
        fig=px.bar(gm2,x="African_Region",y="Score",color="Component",barmode="group",
            color_discrete_sequence=[G,T,GO,CO,PU])
        fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2)
        fig=pf(fig,280); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        ghm=df.groupby("African_Region")[GOV].mean(); ghm.columns=GOV_LBL
        fig=px.imshow(ghm,text_auto=".2f",
            color_continuous_scale=[[0,"#FEF0EC"],[0.5,GO],[1,G]],
            aspect="auto",height=280,zmin=1,zmax=5,labels={"color":"Score"})
        fig=pf(fig,280); fig.update_layout(coloraxis_colorbar=dict(title="Score",len=0.7))
        st.plotly_chart(fig,use_container_width=True)

    sec("KPI Alignment by Organisation Type")
    ctx("Which organisations are best equipped to track KPIs and demonstrate impact to donors?")
    ko=df.groupby("Organization_Type")[["KPI_Alignment_Score","Impact_Measurement_Score"]].mean().reset_index()
    km2=ko.melt(id_vars="Organization_Type",var_name="Metric",value_name="Score")
    km2["Metric"]=km2["Metric"].map({"KPI_Alignment_Score":"KPI Alignment","Impact_Measurement_Score":"Impact Measurement"})
    fig=px.bar(km2,x="Organization_Type",y="Score",color="Metric",barmode="group",
        color_discrete_map={"KPI Alignment":G,"Impact Measurement":T})
    fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2)
    fig=pf(fig,260); fig.update_layout(xaxis_title="",
        yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-25)
    st.plotly_chart(fig,use_container_width=True)

    sec("MEL Action Plan")
    lr=df.groupby("African_Region")["MEL_Score"].mean().idxmin()
    lo=df.groupby("Organization_Type")["KPI_Alignment_Score"].mean().idxmin()
    lc=df[GOV].mean().idxmin().replace("_Score","").replace("_"," ")
    cr(f"<strong>{lr}</strong> has the lowest MEL readiness — prioritise MEL training deployment here first.")
    wn(f"<strong>{lo}</strong> organisations show weakest KPI alignment — provide data templates and M&E coaching.")
    wn(f"<strong>{lc}</strong> is the weakest governance component — include in all governance modules.")
    ok("Respondents with disability have distinct MEL data needs — specify inclusive data collection methods in the MEL framework design.")

    st.divider()
    me=df.groupby("African_Region")[MEL+GOV+["MELAR_Readiness_Score"]].mean().round(2).reset_index()
    dl(me,"MEL Framework Data","afriagri_mel.csv")


# ═══════════════════════════════════════════════════════════════════════════
# P09 — SUSTAINABILITY & PARTNERSHIPS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "09 · Sustainability & Partnerships":
    sec("🔄 Sustainability & Partnerships")
    st.markdown('<div style="font-size:.82rem;color:#666;margin-bottom:12px">AfriAgri\'s long-term success depends on sustainable financing, diverse partnerships, and strong knowledge-sharing systems. This section shows where those foundations are strong and where they need investment.</div>',unsafe_allow_html=True)

    sm=df[SUS].mean()
    c1,c2,c3,c4=st.columns(4)
    for col,lbl,score,bc in zip([c1,c2,c3,c4],SUS_LBL,sm.values,[G,T,GO,PU]):
        grade="Strong" if score>=4.0 else "Moderate" if score>=3.5 else "Needs attention"
        kpi(col,f"{score:.2f}/5",lbl,grade,bc)
    st.markdown("<br>",unsafe_allow_html=True)

    cl,cr2=st.columns(2)
    with cl:
        sec("Sustainability Scores by Region")
        ctx("Which regions have the strongest foundation for long-term success?")
        sr=df.groupby("African_Region")[SUS].mean().reset_index()
        sm2=sr.melt(id_vars="African_Region",var_name="Metric",value_name="Score")
        sm2["Metric"]=sm2["Metric"].map(dict(zip(SUS,SUS_LBL)))
        fig=px.bar(sm2,x="African_Region",y="Score",color="Metric",barmode="group",
            color_discrete_sequence=[G,T,GO,PU])
        fig.add_hline(y=3.5,line_dash="dash",line_color=GR,line_width=1.2)
        fig=pf(fig,290); fig.update_layout(xaxis_title="",
            yaxis=dict(range=[0,5.5],gridcolor="#EEEEEE",title="Score (1–5)"),xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    with cr2:
        sec("Partnership Network Heatmap")
        ctx("Which organisation types partner most with which partner types?")
        ph=df.groupby("Organization_Type")[PART].mean().round(2)
        ph.columns=["Government","Private Sector","Intl. Org","Academia","Farmers Assoc.","CSO"]
        fig=px.imshow(ph,text_auto=".2f",
            color_continuous_scale=[[0,"#F7FBFF"],[0.5,G],[1,T]],
            aspect="auto",height=290,labels={"color":"Rate"})
        fig=pf(fig,290); fig.update_layout(coloraxis_colorbar=dict(title="Partnership\nRate",len=0.7))
        st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        sec("Partnership Diversity Distribution")
        ctx("How many different partner types does each respondent engage with?")
        fig=px.histogram(df,x="Total_Partners",nbins=8,color_discrete_sequence=[G])
        fig.add_vline(x=df["Total_Partners"].mean(),line_dash="dash",line_color=CO,line_width=1.5,
            annotation_text=f"Mean: {df['Total_Partners'].mean():.1f}",annotation_position="top right")
        fig=pf(fig,250); fig.update_layout(showlegend=False,
            xaxis_title="Number of partner types",yaxis=dict(gridcolor="#EEEEEE",title="Respondents"))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sec("Sustainability vs Vulnerability")
        ctx("High sustainability + low vulnerability = ideal position. Regions bottom-left need support.")
        sv=df.groupby("African_Region").agg(
            Sustainability=("Sustainability_Score","mean"),
            Vulnerability=("Vulnerability_Index","mean"),
            Count=("Respondent_ID","count")).reset_index()
        fig=px.scatter(sv,x="Sustainability",y="Vulnerability",size="Count",
            color="African_Region",text="African_Region",color_discrete_map=REGCOL,
            labels={"Sustainability":"Sustainability (1–5)","Vulnerability":"Vulnerability (1–5)"})
        fig.update_traces(textposition="top center",textfont_size=9)
        fig=pf(fig,250); fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    sec("AI Resource Allocation — Where AfriAgri Should Deploy First")
    ctx("Composite priority score combining vulnerability, empowerment gaps, intervention urgency, and disability rates. Higher = deploy here first.")
    al=df.groupby("African_Region").agg(
        Respondents=("Respondent_ID","count"),
        Vulnerability=("Vulnerability_Index","mean"),
        WAEI=("WAEI_Score","mean"),
        Critical_Rate=("Intervention_Priority",lambda x:(x=="Critical").mean()*100),
        Disability_Rate=("Disability_Flag","mean"),
        Sustainability=("Sustainability_Score","mean"),
    ).round(2).reset_index()
    al["Priority_Score"]=(al["Vulnerability"]*0.35+(100-al["WAEI"])/100*5*0.30+
        al["Critical_Rate"]/100*5*0.25+al["Disability_Rate"]*5*0.10).round(3)
    al=al.sort_values("Priority_Score",ascending=False).reset_index(drop=True)
    al.index=al.index+1
    al["Action"]=al["Priority_Score"].apply(
        lambda x:"🔴 Immediate deployment" if x>=3.5 else "🟡 High priority" if x>=3.0 else "🟢 Routine monitoring")
    st.dataframe(
        al[["African_Region","Priority_Score","Vulnerability","WAEI","Critical_Rate","Disability_Rate","Sustainability","Action"]].rename(
            columns={"African_Region":"Region","Priority_Score":"Priority Score",
                "Vulnerability":"Vuln.","Critical_Rate":"Critical%","Disability_Rate":"Disability Rate"}),
        use_container_width=True,
        column_config={
            "WAEI":st.column_config.ProgressColumn("WAEI",min_value=0,max_value=100,format="%.1f"),
            "Priority Score":st.column_config.NumberColumn(format="%.3f"),
            "Critical%":st.column_config.NumberColumn(format="%.1f%%"),
        })

    lr2=df.groupby("African_Region")["Sustainability_Score"].mean().idxmin()
    bp=df.groupby("African_Region")["Total_Partners"].mean().idxmax()
    ta=al.iloc[0]["African_Region"]
    st.markdown("<br>",unsafe_allow_html=True)
    cr(f"<strong>{ta}</strong> scores highest on allocation priority — deploy here first.")
    wn(f"<strong>{lr2}</strong> has the weakest sustainability foundation — funding diversification support needed urgently.")
    ok(f"<strong>{bp}</strong> has the most diverse partnership networks — use as a model for other regions.")

    st.divider()
    se2=df.groupby(["African_Region","Organization_Type"])[SUS+PART+["Total_Partners"]].mean().round(2).reset_index()
    dl(se2,"Sustainability & Partnerships","afriagri_sustainability.csv")
