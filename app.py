import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Mental Health Risk Assessment Agent", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("risk_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

HIGH_RISK_KEYWORDS = [
    "end my life", "kill myself", "want to die", "no reason to live",
    "better off without me", "can't go on", "give up completely",
    "hurt myself", "self harm", "ending it all"
]

CRISIS_RESOURCES = """
**988 Suicide & Crisis Lifeline (US):** call or text 988
**Crisis Text Line:** text HOME to 741741
**International resources:** https://www.iasp.info/resources/Crisis_Centres/
"""

if "records" not in st.session_state:
    st.session_state.records = []

def keyword_flag(text):
    t = text.lower()
    return any(k in t for k in HIGH_RISK_KEYWORDS)

def analyze_text(text, user_id):
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    idx = np.argmax(proba)
    risk_level = classes[idx]
    confidence = float(proba[idx])

    flag = keyword_flag(text)
    if flag and risk_level != "high":
        risk_level = "high"
        confidence = max(confidence, 0.85)

    return {
        "user_id": user_id,
        "text": text,
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "keyword_flag": flag,
        "timestamp": datetime.now().isoformat(),
    }

st.title("🧠 Mental Health Risk Assessment Agent")
st.caption("A support/triage tool. Not a diagnostic system — high-risk alerts should always be reviewed by a qualified professional.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Text")
    user_id = st.text_input("User ID (optional)", value="anonymous")
    text_input = st.text_area("Enter text to analyze", height=150,
                               placeholder="Type or paste the message here...")

    if st.button("Analyze", type="primary"):
        if text_input.strip():
            result = analyze_text(text_input, user_id)
            st.session_state.records.append(result)

            risk = result["risk_level"]
            if risk == "high":
                st.error(f"🚨 HIGH RISK detected (confidence: {result['confidence']})")
                st.markdown("**Crisis Resources:**")
                st.markdown(CRISIS_RESOURCES)
            elif risk == "moderate":
                st.warning(f"⚠️ MODERATE RISK detected (confidence: {result['confidence']})")
                st.info("Recommend a follow-up check-in with this user.")
            else:
                st.success(f"✅ Low risk (confidence: {result['confidence']})")
        else:
            st.warning("Please enter some text to analyze.")

with col2:
    st.subheader("Session Summary")
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.metric("Total analyzed", len(df))
        st.metric("High risk cases", int((df["risk_level"] == "high").sum()))
        st.metric("Moderate risk cases", int((df["risk_level"] == "moderate").sum()))
    else:
        st.write("No records yet.")

st.divider()
st.subheader("Risk Assessment Records")

if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df[["timestamp", "user_id", "risk_level", "confidence", "keyword_flag"]],
                 use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download records as CSV", csv, "risk_records.csv", "text/csv")
else:
    st.write("No records logged yet.")
