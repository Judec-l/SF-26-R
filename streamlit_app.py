import streamlit as st
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Particle Tracking Velocimetry",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling (Clean & Professional)
# --------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 18px;
            color: gray;
            margin-top: 0px;
        }
        .section-header {
            font-size: 22px;
            font-weight: 600;
            margin-top: 30px;
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 14px;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">Particle Tracking Velocimetry Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Configure parameters and run displacement field extraction</div>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Layout Columns
# --------------------------------------------------
left_col, right_col = st.columns([2, 1])

# --------------------------------------------------
# LEFT COLUMN — File & Parameter Settings
# --------------------------------------------------
with left_col:

    st.markdown('<div class="section-header">1. Select Input Data</div>', unsafe_allow_html=True)

    input_folder = st.text_input(
        "Input Image Directory",
        placeholder="e.g. /data/test_sequence/"
    )

    output_folder = st.text_input(
        "Output Directory",
        placeholder="e.g. /results/output_run_01/"
    )

    st.markdown('<div class="section-header">2. Tracking Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        detection_threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.01
        )

    with col2:
        max_displacement = st.number_input(
            "Max Displacement (px)",
            min_value=1,
            value=15
        )

    with col3:
        min_trajectory = st.number_input(
            "Min Trajectory Length",
            min_value=1,
            value=5
        )

    st.divider()

    run_button = st.button("🚀 Run Particle Tracking", use_container_width=True)

# --------------------------------------------------
# RIGHT COLUMN — Summary Panel
# --------------------------------------------------
with right_col:

    st.markdown('<div class="section-header">Configuration Summary</div>', unsafe_allow_html=True)

    st.info(f"""
    **Input Directory:**  
    {input_folder if input_folder else "Not selected"}

    **Output Directory:**  
    {output_folder if output_folder else "Not selected"}

    **Detection Threshold:** {detection_threshold}  
    **Max Displacement:** {max_displacement} px  
    **Min Trajectory Length:** {min_trajectory}
    """)

# --------------------------------------------------
# Simulated Run Output (UI Only)
# --------------------------------------------------
if run_button:
    st.divider()
    st.markdown("### Processing Status")

    if not input_folder or not output_folder:
        st.error("Please specify both input and output directories.")
    else:
        st.success("Configuration valid. Ready to execute main_ptv_runner.py")
        st.progress(100)
        st.write("Processing complete. Results saved to output directory.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown('<div class="footer">Particle Tracking Research Interface | 2026 Science Project</div>', unsafe_allow_html=True)
