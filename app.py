"""
Lead Conversion Prediction — Streamlit showcase app
Knovista live project: predicts whether a lead will convert into a customer.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, accuracy_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DT_FMT = "%d-%m-%Y %H:%M"

st.set_page_config(page_title="Lead Conversion Prediction", page_icon="🎯", layout="wide")

# --------------------------------------------------------------------------- #
# Feature engineering (same logic as the notebook, kept in one place so the
# training data and any new lead the user enters go through identical steps)
# --------------------------------------------------------------------------- #

def campaign_to_course(name):
    n = str(name).lower()
    rules = [("python", "Python"), ("data analytics", "Data Analytics"), ("-da", "Data Analytics"),
             ("business accounting", "Business Accounting"), ("ba ", "Business Accounting"),
             ("-ba", "Business Accounting"), ("digital marketing", "Digital Marketing"),
             ("dm", "Digital Marketing"), ("ui/ux", "UI/UX"), ("php", "PHP"), ("sap", "SAP"),
             ("performance", "Performance Marketing")]
    for key, course in rules:
        if key in n:
            return course
    return "Other"


def source_group(name):
    if name in ("INSTAGRAM", "FACEBOOK_ADS", "MOVE_LEAD", "WALK_IN_LEAD"):
        return name
    if name == "full leads.csv":
        return "FULL_LEADS_FILE"
    return "OTHER_FILE_UPLOAD"


# Friendly labels for the UI, and a representative raw source_name for each
# group (the model only ever sees source_group, never the raw file name, so
# any representative value in the right group gives an identical prediction).
SOURCE_GROUP_LABELS = {
    "INSTAGRAM": "Instagram",
    "FACEBOOK_ADS": "Facebook Ads",
    "MOVE_LEAD": "Moved from another list",
    "WALK_IN_LEAD": "Walk-in",
    "FULL_LEADS_FILE": "Bulk leads file",
    "OTHER_FILE_UPLOAD": "Other file upload",
}
SOURCE_GROUP_SAMPLE_NAME = {
    "INSTAGRAM": "INSTAGRAM",
    "FACEBOOK_ADS": "FACEBOOK_ADS",
    "MOVE_LEAD": "MOVE_LEAD",
    "WALK_IN_LEAD": "WALK_IN_LEAD",
    "FULL_LEADS_FILE": "full leads.csv",
    "OTHER_FILE_UPLOAD": "other_upload.xlsx",
}


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    txt = lambda col: d[col].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
    out = pd.DataFrame(index=d.index)

    out["campaign_name"] = txt("campaign_name")
    out["course"] = out["campaign_name"].map(campaign_to_course)
    out["is_refurbished"] = out["campaign_name"].str.contains("refurbished", case=False).astype(int)
    out["is_no_stipend"] = out["campaign_name"].str.contains("stipend", case=False).astype(int)
    out["source_type"] = txt("source_type")
    out["source_group"] = txt("source_name").map(source_group)
    out["assigned_to"] = txt("assigned_to").str.replace(r"\s+", " ", regex=True).str.title()

    created = pd.to_datetime(d["creation_date"], format=DT_FMT, errors="coerce")
    out["created_hour"] = created.dt.hour.fillna(12)
    out["created_dow"] = created.dt.dayofweek.fillna(0)
    out["is_weekend"] = (created.dt.dayofweek >= 5).fillna(False).astype(int)

    first_call = pd.to_datetime(d["first_call_attempt"], format=DT_FMT, errors="coerce")
    out["hours_to_first_call"] = ((first_call - created).dt.total_seconds() / 3600).clip(lower=0)
    return out


CAT_COLS = ["campaign_name", "course", "source_type", "source_group", "assigned_to"]
NUM_COLS = ["is_refurbished", "is_no_stipend", "created_hour", "created_dow",
            "is_weekend", "hours_to_first_call"]

# --------------------------------------------------------------------------- #
# Data + model loading (cached so the app only trains once per session)
# --------------------------------------------------------------------------- #

def make_demo_data(n: int = 8000, seed: int = 7) -> pd.DataFrame:
    """Synthetic leads with the same columns as Unified_Leads.csv.
    Nothing here comes from Knovista: names, contacts and outcomes are all made up,
    so the public demo never exposes real lead details."""
    rng = np.random.default_rng(seed)
    campaigns = {
        "Python Campaign": 0.4, "Python No Stipend Campaign": -0.2,
        "Data Analytics Campaign": 0.6, "Data Analytics Refurbished Campaign": -0.4,
        "Business Accounting Campaign": 0.3, "Digital Marketing Campaign": 0.0,
        "Digital Marketing No Stipend Campaign": -0.5, "UI/UX Campaign": -0.1,
        "PHP Campaign": -0.3, "SAP Campaign": 0.2,
    }
    sources = {  # (source_type, source_name): effect
        ("INSTAGRAM", "INSTAGRAM"): 0.2, ("FACEBOOK_ADS", "FACEBOOK_ADS"): 0.0,
        ("FILE_UPLOAD", "bulk_upload_1.xlsx"): -0.5, ("FILE_UPLOAD", "full leads.csv"): -0.3,
        ("MOVE_LEAD", "MOVE_LEAD"): 0.1, ("WALK_IN_LEAD", "WALK_IN_LEAD"): 1.2,
    }
    reps = {"Rep A": 0.5, "Rep B": 0.2, "Rep C": 0.0, "Rep D": -0.2, "Rep E": -0.4, "Rep F": 0.1}

    camp_names = rng.choice(list(campaigns), n)
    src_keys = list(sources)
    src_idx = rng.choice(len(src_keys), n, p=[0.30, 0.30, 0.15, 0.10, 0.10, 0.05])
    rep_names = rng.choice(list(reps), n)

    start = pd.Timestamp("2025-06-01")
    created = start + pd.to_timedelta(rng.integers(0, 250 * 24 * 60, n), unit="m")
    called = rng.random(n) > 0.25
    hours = rng.exponential(10, n)

    z = (-1.7
         + 2.2 * np.array([campaigns[c] for c in camp_names])
         + 2.2 * np.array([sources[src_keys[i]] for i in src_idx])
         + 2.2 * np.array([reps[r] for r in rep_names])
         - 0.06 * np.where(called, hours, 40)
         - 1.0 * (~called))
    converted = rng.random(n) < 1 / (1 + np.exp(-z))

    resolved = rng.random(n) < 0.65
    stage = np.where(resolved, np.where(converted, "CONVERTED", "LOST"),
                     rng.choice(["Prospect", "Follow Up Required"], n))
    fmt = lambda s: s.dt.strftime(DT_FMT)
    created_s = pd.Series(created)
    first_call = pd.Series(created + pd.to_timedelta(hours, unit="h")).where(called)

    return pd.DataFrame({
        "Name": [f"Demo Lead {i:05d}" for i in range(n)],
        "contact_number": [f"DEMO-{i:05d}" for i in range(n)],
        "Contact_email": [f"demo{i:05d}@example.invalid" for i in range(n)],
        "campaign_name": camp_names,
        "source_type": [src_keys[i][0] for i in src_idx],
        "source_name": [src_keys[i][1] for i in src_idx],
        "assigned_to": rep_names,
        "creation_date": fmt(created_s),
        "first_call_attempt": fmt(first_call).where(called),
        "lead_stage": stage,
    })


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.strip()
    df["assigned_to"] = df["assigned_to"].str.replace(r"\s+", " ", regex=True).str.title()
    df["creation_dt"] = pd.to_datetime(df["creation_date"], format=DT_FMT, errors="coerce")

    labeled = df[df["lead_stage"].isin(["CONVERTED", "LOST"])].drop_duplicates(subset="contact_number").copy()
    labeled["converted"] = (labeled["lead_stage"] == "CONVERTED").astype(int)
    open_leads = df[~df["lead_stage"].isin(["CONVERTED", "LOST"])].copy()
    return df, labeled, open_leads


@st.cache_data(show_spinner="Loading and cleaning the leads data...")
def load_data(path):
    return prepare_data(pd.read_csv(path))


@st.cache_data(show_spinner="Generating synthetic demo leads...")
def load_demo():
    return prepare_data(make_demo_data())


@st.cache_resource(show_spinner="Training the model (Random Forest, this takes a few seconds)...")
def train_model(labeled: pd.DataFrame):
    X = build_features(labeled)
    y = labeled["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_COLS),
    ])
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocess), ("model", model)]).fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred_test),
        "Precision": precision_score(y_test, pred_test, zero_division=0),
        "Recall": recall_score(y_test, pred_test),
        "F1": f1_score(y_test, pred_test),
        "ROC-AUC": roc_auc_score(y_test, proba_test),
        "PR-AUC": average_precision_score(y_test, proba_test),
    }
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test)

    final_model = clone(pipe).fit(X, y)  # refit on all labeled data for real predictions
    return final_model, metrics, (fpr, tpr), cm


def priority_tier(p, threshold=0.5):
    if p >= threshold:
        return "🔴 HIGH"
    if p >= threshold / 2:
        return "🟡 MEDIUM"
    return "🟢 LOW"


# --------------------------------------------------------------------------- #
# Sidebar: data source
# --------------------------------------------------------------------------- #

st.sidebar.title("🎯 Lead Conversion Prediction")
st.sidebar.caption("Knovista live project — Machine Learning classification")
data_file = st.sidebar.file_uploader("Unified_Leads.csv", type="csv")
default_path = "Unified_Leads.csv"

DEMO_MODE = False
if data_file is not None:
    df, labeled, open_leads = load_data(data_file)
else:
    try:
        df, labeled, open_leads = load_data(default_path)
    except FileNotFoundError:
        DEMO_MODE = True
        df, labeled, open_leads = load_demo()
        st.sidebar.info("Demo mode: using synthetic leads. Upload the real CSV above to use real data.")

model, metrics, roc_data, cm = train_model(labeled)

page = st.sidebar.radio("Go to", ["Overview", "Predict a lead", "Model performance", "Score open leads"])

if DEMO_MODE:
    st.warning(
        "**Demo mode:** this app is running on **synthetic, made-up leads** so no real "
        "lead details are public. The numbers below are not Knovista's real results. "
        "The real analysis is in the project notebook on GitHub."
    )

# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #

if page == "Overview":
    st.title("Lead Conversion Prediction")
    st.markdown(
        "Predicts whether a lead will convert into a customer, using lead source, "
        "campaign, assigned salesperson, and timing — so the sales team can prioritize follow-ups."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total leads", f"{len(df):,}")
    c2.metric("Labeled (Converted/Lost)", f"{len(labeled):,}")
    c3.metric("Still open", f"{len(open_leads):,}")
    c4.metric("Conversion rate", f"{labeled['converted'].mean():.1%}")

    st.subheader("Conversion rate by course")
    X_all = build_features(labeled)
    rate = (X_all.assign(converted=labeled["converted"].values)
            .groupby("course")["converted"].mean().sort_values(ascending=False) * 100)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(rate.index, rate.values, color="#2563eb")
    ax.set_xlabel("Conversion rate (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("Conversion rate by source type")
    rate2 = (X_all.assign(converted=labeled["converted"].values)
             .groupby("source_type")["converted"].mean().sort_values(ascending=False) * 100)
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.bar(rate2.index, rate2.values, color="#f59e0b")
    ax2.set_ylabel("Conversion rate (%)")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig2)

    if not DEMO_MODE:
        st.info(
            "⚠️ **Honest caveat:** a robustness check (time-based split, and removing the "
            "top-converting salesperson) showed most of this signal comes from one salesperson/"
            "campaign combination early in the data, not generalizable lead attributes. "
            "See the project notebook for the full analysis."
        )

# --------------------------------------------------------------------------- #
# Page: Predict a lead
# --------------------------------------------------------------------------- #

elif page == "Predict a lead":
    st.title("Predict a new lead")
    st.caption("Enter a lead's details to get a conversion probability and priority tier.")

    # Clean course -> campaign hierarchy, built once per session. Course alone
    # carries almost all of campaign_name's predictive signal (verified on the
    # held-out test set: dropping campaign_name moves ROC-AUC by < 0.001), so
    # the form only asks for Course; a representative campaign for that course
    # is picked automatically behind the scenes for the model to consume.
    course_map = (df[["campaign_name"]].dropna().assign(course=df["campaign_name"].map(campaign_to_course)))
    courses = sorted(course_map["course"].unique())
    representative_campaign = (course_map.groupby("course")["campaign_name"]
                               .agg(lambda s: s.value_counts().idxmax()).to_dict())

    # Which source groups are actually present in the data, with friendly labels.
    present_groups = sorted(df["source_name"].dropna().map(source_group).unique())
    group_labels = [SOURCE_GROUP_LABELS.get(g, g) for g in present_groups]

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            course = st.selectbox("Course", courses)
            source_type = st.selectbox("Source type", sorted(df["source_type"].dropna().unique()))
            source_label = st.selectbox("Source", group_labels)
        with c2:
            assigned_to = st.selectbox("Assigned to", sorted(df["assigned_to"].dropna().unique()))
            was_called = st.checkbox("Already called?", value=True)
            hours_to_call = st.slider("Hours until first call", 0, 72, 4) if was_called else None
        submitted = st.form_submit_button("Predict")

    if submitted:
        now = pd.Timestamp.now()
        creation_date = now.strftime(DT_FMT)
        first_call_attempt = (now + pd.Timedelta(hours=hours_to_call)).strftime(DT_FMT) if was_called else None

        # Map the friendly source label back to a raw source_name the feature
        # pipeline recognizes; build_features() will re-derive the same clean
        # source_group from it, so the prediction is identical either way.
        source_group_key = present_groups[group_labels.index(source_label)]
        source_name = SOURCE_GROUP_SAMPLE_NAME.get(source_group_key, source_group_key)
        campaign_name = representative_campaign[course]

        row = pd.DataFrame([{
            "campaign_name": campaign_name, "source_type": source_type, "source_name": source_name,
            "assigned_to": assigned_to, "creation_date": creation_date,
            "first_call_attempt": first_call_attempt,
        }])
        proba = float(model.predict_proba(build_features(row))[:, 1][0])

        st.metric("Conversion probability", f"{proba:.1%}")
        st.markdown(f"### Priority: {priority_tier(proba)}")
        st.progress(min(proba, 1.0))

# --------------------------------------------------------------------------- #
# Page: Model performance
# --------------------------------------------------------------------------- #

elif page == "Model performance":
    st.title("Model performance")
    st.caption("Random Forest, evaluated on a held-out 20% test split.")

    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curve")
        fpr, tpr = roc_data
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="#2563eb", label=f"AUC = {metrics['ROC-AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)
    with c2:
        st.subheader("Confusion matrix")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax2.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Lost", "Converted"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Lost", "Converted"])
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        st.pyplot(fig2)

    st.warning(
        "Accuracy alone is misleading here — always predicting 'Lost' already scores "
        f"~{1 - labeled['converted'].mean():.0%}. Precision/Recall/ROC-AUC tell the real story."
    )

# --------------------------------------------------------------------------- #
# Page: Score open leads
# --------------------------------------------------------------------------- #

elif page == "Score open leads":
    st.title("Score open leads")
    st.caption("Ranks every currently open lead (Prospect / Follow Up Required) by conversion probability.")

    open_feats = build_features(open_leads)
    scored = open_leads.copy()
    scored["conversion_probability"] = model.predict_proba(open_feats)[:, 1]
    scored["priority"] = scored["conversion_probability"].map(priority_tier)
    scored = scored.sort_values("conversion_probability", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("High priority", int((scored["priority"] == "🔴 HIGH").sum()))
    c2.metric("Medium priority", int((scored["priority"] == "🟡 MEDIUM").sum()))
    c3.metric("Low priority", int((scored["priority"] == "🟢 LOW").sum()))

    display_cols = ["Name", "contact_number", "campaign_name", "source_type",
                    "assigned_to", "conversion_probability", "priority"]
    display_cols = [c for c in display_cols if c in scored.columns]
    st.dataframe(scored[display_cols].head(200), use_container_width=True)

    st.download_button(
        "⬇ Download full ranked call list (CSV)",
        scored[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="scored_open_leads.csv",
        mime="text/csv",
    )
