"""
=============================================================================
 Car Price Prediction — Streamlit Web Application
=============================================================================
 A clean, modern UI for predicting car selling prices using
 the trained machine learning model.

 Usage:
   streamlit run app/app.py
=============================================================================
"""

import os
import sys
import streamlit as st
import numpy as np

# Add project root to path so we can import src.predict
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_price, load_model


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for Premium Look ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }

    /* ── Header styling ── */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* ── Card container ── */
    .card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* ── Result display ── */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        animation: fadeIn 0.5s ease-in;
    }
    .result-label {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .result-price {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    .result-unit {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(102, 126, 234, 0.4), transparent);
        margin: 1.5rem 0;
    }

    /* ── Stacked button ── */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🚗 Car Price Prediction</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Predict the resale value of your car using Machine Learning</p>',
    unsafe_allow_html=True,
)

# ── Check if model exists & Load Names ───────────────────────────────────────
model_path = os.path.join(PROJECT_ROOT, "model", "model.pkl")
if not os.path.exists(model_path):
    st.error(
        "⚠️ **Model not found!** Please run `python train.py` first to train and save the model."
    )
    st.stop()
else:
    try:
        artifact = load_model(model_path)
        # Title case for better UI presentation
        car_models = sorted([str(name).title() for name in artifact.get("car_names", ["Corolla", "Civic", "City"])])
    except Exception:
        car_models = ["Corolla", "Civic", "City"]


# ── Input Form ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Car Details")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox(
        "🚘 Car Model",
        options=car_models,
        index=0 if "Corolla" not in car_models else car_models.index("Corolla"),
        help="Select the model of the car",
    )

    year = st.number_input(
        "📅 Year of Purchase",
        min_value=2000,
        max_value=2026,
        value=2018,
        step=1,
        help="The year the car was originally purchased",
    )

    present_price = st.number_input(
        "💰 Present Price (Lakhs PKR)",
        min_value=0.1,
        max_value=1000.0,
        value=60.0,
        step=0.1,
        format="%.2f",
        help="Current showroom price of the car in Pakistani Lakhs (used for prediction)",
    )

    kms_driven = st.number_input(
        "🛣️ Kilometres Driven",
        min_value=0,
        max_value=500000,
        value=30000,
        step=1000,
        help="Total distance driven in kilometres",
    )

    owner = st.selectbox(
        "👤 Number of Previous Owners",
        options=[0, 1, 2, 3],
        index=0,
        help="How many people owned this car before",
    )

with col2:
    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        options=["Petrol", "Diesel", "CNG"],
        index=0,
        help="Type of fuel the car uses",
    )

    seller_type = st.selectbox(
        "🏪 Seller Type",
        options=["Dealer", "Individual"],
        index=0,
        help="Are you a dealer or individual seller?",
    )

    transmission = st.selectbox(
        "⚙️ Transmission",
        options=["Manual", "Automatic"],
        index=0,
        help="Type of transmission",
    )

    # Info card
    st.markdown("")
    st.markdown("")
    st.info(
        "💡 **Tip:** The model was trained on Pakistani car market data. "
        "Results are most accurate for cars from 2003–2018."
    )


# ── Predict Button ───────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

if st.button("🔮 Predict Selling Price", use_container_width=True):
    with st.spinner("Calculating..."):
        try:
            predicted_price = predict_price(
                car_name=car_name.lower(),
                year=year,
                present_price=present_price,
                kms_driven=kms_driven,
                fuel_type=fuel_type,
                seller_type=seller_type,
                transmission=transmission,
                owner=owner,
                model_path=model_path,
            )

            # Predicted price is already in PKR Lakhs
            predicted_pkr_lakhs = predicted_price
            predicted_pkr = predicted_pkr_lakhs * 100000  # full PKR value

            # Display result
            if predicted_pkr >= 10000000:  # 1 Crore+
                display_price = f"₨ {predicted_pkr / 10000000:.2f} Crore"
                display_unit = "Pakistani Rupees (Crore)"
            elif predicted_pkr >= 100000:  # 1 Lakh+
                display_price = f"₨ {predicted_pkr_lakhs:.2f} Lakh"
                display_unit = "Pakistani Rupees (Lakh)"
            else:
                display_price = f"₨ {predicted_pkr:,.0f}"
                display_unit = "Pakistani Rupees"

            st.markdown(
                f"""
                <div class="result-box">
                    <p class="result-label">Estimated Selling Price</p>
                    <p class="result-price">{display_price}</p>
                    <p class="result-unit">{display_unit}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show input summary
            with st.expander("📊 Input Summary", expanded=False):
                summary_data = {
                    "Feature": [
                        "Car Model", "Year", "Present Price", "Kms Driven",
                        "Fuel Type", "Seller Type", "Transmission", "Owners",
                    ],
                    "Value": [
                        car_name, year, f"₨{present_price:.2f}L", f"{kms_driven:,} km",
                        fuel_type, seller_type, transmission, owner,
                    ],
                }
                st.table(summary_data)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Built with ❤️ using Streamlit & Scikit-learn  •  "
    "Car Price Prediction ML Project"
    "</div>",
    unsafe_allow_html=True,
)
