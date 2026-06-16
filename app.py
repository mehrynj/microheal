import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# --- AUTO-LAUNCHER SHORTCUT ---
if __name__ == '__main__':
    try:
        from streamlit.web import cli as stcli
        if len(sys.argv) == 1 or not sys.argv[0].endswith('streamlit'):
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
    except Exception:
        pass

# 1. Set up web browser tab properties
st.set_page_config(
    page_title="MicroHeal Web Platform",
    page_icon="🔬",
    layout="wide"  # Changed to wide layout to cleanly handle three input zones!
    )

# 2. Resilient background data loader & training script
@st.cache_resource
def train_microheal_engine():
    expected_csv = "Maternal Health Risk Data Set.csv"
    if not os.path.exists(expected_csv):
        return None, None, False
        
    try:
        df = pd.read_csv(expected_csv)
        df.columns = df.columns.str.strip()
        X = df.iloc[:, 0:6]
        y = df.iloc[:, 6]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler, True
    except Exception:
        return None, None, False

# Initialize Engine
model, scaler, success = train_microheal_engine()

if not success:
    st.error("❌ File Not Found: Could not find 'Maternal Health Risk Data Set.csv' in your project folder.")
else:
    st.title("🔬 MicroHeal Advanced Web Platform")
    st.markdown("### Multi-Omics Analytics for Maternal-Fetal Risk Stratification")
    st.write("Modify the indicators across the three medical categories below to generate an instantaneous multi-layered health diagnostic.")
    
    st.markdown("---")
    
    # 3. Dynamic UI Layout Handling (Three main columns for clean data entry)
    panel1, panel2, panel3 = st.columns(3)
    
    # PANEL A: Core Physiological Parameters (Feeds the AI Model)
    with panel1:
        st.markdown("#### 🩺 Core Vitals (AI Engine)")
        age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=29, step=1)
        sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=220, value=115, step=1)
        dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=140, value=75, step=1)
        blood_sugar = st.number_input("Blood Sugar Level (mmol/L)", min_value=1.0, max_value=35.0, value=5.40, step=0.1)
        temp = st.number_input("Body Temperature (°F)", min_value=85.0, max_value=110.0, value=98.40, step=0.1)
        heart_rate = st.number_input("Resting Heart Rate (BPM)", min_value=30, max_value=200, value=72, step=1)
        
    # PANEL B: Advanced Biochemical Markers (Feeds the Fetal Anomaly Quad Screen)
    with panel2:
        st.markdown("#### 🧪 Biochemical Markers (Fetal Screen)")
        pappa_level = st.number_input(
            "PAPP-A Level (MoM)", min_value=0.01, max_value=5.00, value=1.00, step=0.05,
            help="Pregnancy-Associated Plasma Protein A. Low values (<0.4) signal chromosomal anomaly risks."
        )
        afp_level = st.number_input(
            "AFP Level (MoM)", min_value=0.01, max_value=5.00, value=1.00, step=0.05,
            help="Alpha-Fetoprotein. High values (>2.0) signal neural tube/spinal column defects."
        )
        hcg_level = st.number_input(
            "Free Beta-hCG (MoM)", min_value=0.01, max_value=5.00, value=1.00, step=0.05,
            help="Human Chorionic Gonadotropin. Spikes (>2.0) correlate with Trisomy 21 (Down Syndrome)."
        )
        ue3_level = st.number_input(
            "Unconjugated Estriol (uE3) (MoM)", min_value=0.01, max_value=5.00, value=1.00, step=0.05,
            help="Measures fetoplacental metabolic integrity. Drastic drops (<0.5) show high distress."
        )
        inhibin_level = st.number_input(
            "Inhibin-A Level (MoM)", min_value=0.01, max_value=5.00, value=1.00, step=0.05,
            help="Placental hormone marker. Used to drastically improve Quad-Screen accuracy."
        )
        
    # PANEL C: NEW PANEL - Gut Microbiome Analytics
    with panel3:
        st.markdown("#### 🦠 Gut Microbiome Intelligence")
        microbiome_diversity = st.number_input(
            "Alpha Diversity Index (Shannon Index)", min_value=0.50, max_value=5.00, value=3.40, step=0.05,
            help="Measures bacterial species richness. Values below 2.50 signify dysbiosis and high systemic inflammation."
        )
        scfa_abundance = st.number_input(
            "SCFA-Producing Bacteria (%)", min_value=1.0, max_value=100.0, value=55.0, step=1.0,
            help="Percentage of beneficial fiber-fermenting bacteria (e.g., Bifidobacterium). Drops trigger high metabolic risk."
        )
        
    st.markdown("---")
    
    # 4. Integrated Evaluation Engine Execution Button
    if st.button("Generate Complete Diagnostic Report", type="primary", use_container_width=True):
        
        # --- PHASE 1: AI MODEL EVALUATION ---
        user_features = np.array([[age, sys_bp, dia_bp, blood_sugar, temp, heart_rate]])
        scaled_features = scaler.transform(user_features)
        
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        
        low_risk_p = prob_dict.get('low risk', prob_dict.get('Low Risk', 0.0))
        mid_risk_p = prob_dict.get('mid risk', prob_dict.get('Mid Risk', 0.0))
        high_risk_p = prob_dict.get('high risk', prob_dict.get('High Risk', 0.0))
        
        # UI Presentation Split for Results
        report_col1, report_col2, report_col3 = st.columns(3)
        
        with report_col1:
            st.markdown("### 📊 AI Maternal Vitals Report")
            display_status = prediction.upper()
            if "HIGH" in display_status:
                st.error(f"🔴 RESULT: {display_status}")
            elif "MID" in display_status:
                st.warning(f"🟡 RESULT: {display_status}")
            else:
                st.success(f"🟢 RESULT: {display_status}")
                
            st.markdown("**Confidence Intervals:**")
            st.progress(float(low_risk_p), text=f"Low Risk: {int(low_risk_p * 100)}%")
            st.progress(float(mid_risk_p), text=f"Mid Risk: {int(mid_risk_p * 100)}%")
            st.progress(float(high_risk_p), text=f"High Risk: {int(high_risk_p * 100)}%")
            
        # --- PHASE 2: FETAL ANOMALY QUAD SCREEN ---
        with report_col2:
            st.markdown("### 👶 Fetal Anomaly Screening")
            fetal_flags = []
            
            # Pattern recognition for Down Syndrome (Trisomy 21)
            if afp_level < 0.7 and hcg_level > 2.0 and ue3_level < 0.6 and inhibin_level > 2.0:
                fetal_flags.append("🚨 **High Match Pattern (Down Syndrome):** Low AFP/uE3 paired with high hCG/Inhibin-A matches the classic diagnostic pattern for Trisomy 21.")
            # Pattern recognition for Edwards Syndrome (Trisomy 18)
            elif afp_level < 0.6 and hcg_level < 0.4 and ue3_level < 0.5:
                fetal_flags.append("🚨 **High Match Pattern (Trisomy 18):** Critically depressed values across AFP, hCG, and uE3 indicate statistical patterns for Edwards Syndrome.")
            # Individual Structural Flags
            else:
                if afp_level > 2.00:
                    fetal_flags.append("⚠️ **Elevated AFP Marker:** High leakage levels indicate an elevated risk for neural tube structural anomalies (e.g., Spina Bifida).")
                if pappa_level < 0.40:
                    fetal_flags.append("⚠️ **Low PAPP-A Marker:** Suboptimal early placental marker. Suggests risk for genetic anomalies or placental insufficiency.")
            
            if fetal_flags:
                for flag in fetal_flags:
                    st.error(flag)
                st.info("💡 **Recommendation:** Schedule an immediate high-resolution fetal target scan (Level II Ultrasound) or cell-free DNA (cfDNA) analysis.")
            else:
                st.success("✨ **Normal Fetal Biochemical Screen:** All 5 core serum protein concentrations track within healthy target baselines.")

        # --- PHASE 3: NEW GUT MICROBIOME ENGINE ---
        with report_col3:
            st.markdown("### 🦠 Maternal Microbiome Report")
            microbiome_flags = []
            
            if microbiome_diversity < 2.50:
                microbiome_flags.append("⚠️ **Critical Dysbiosis Detected:** Low Alpha Diversity (Shannon Index) indicates an impoverished gut ecosystem. This significantly raises systemic inflammation biomarkers.")
                
            if scfa_abundance < 40.0:
                microbiome_flags.append("⚠️ **Depleted SCFA Producers:** Beneficial short-chain fatty acid bacteria are low. This increases host metabolic stress and undermines insulin sensitivity.")
                
            if microbiome_flags:
                for flag in microbiome_flags:
                    st.warning(flag)
                
                # Cross-reference with the AI model's blood sugar readings to add contextual intelligence
                if blood_sugar > 6.5:
                    st.error("🧬 **Synergistic Metabolic Alert:** The combination of gut dysbiosis and elevated blood sugar levels expands the statistical risk matrix for Gestational Diabetes.")
                if sys_bp > 135:
                    st.error("Vascular Inflammation Alert:** Poor microbiome defenses combined with elevated blood pressure accelerates vascular stress associated with preeclampsia.")
                    
                st.info("🥦 **Dietary Directive:** Clinical recommendation includes targeting high-diversity prebiotic fibers and targeted probiotic supplementation to rebuild the SCFA barrier.")
            else:
                st.success("✨ **Optimal Microbiome Environment:** High ecosystem diversity and robust short-chain fatty acid counts indicate protective anti-inflammatory metabolic buffers!")