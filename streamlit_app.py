import streamlit as st
import requests

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Insurance Claim Prediction")
st.write("MLOps-powered Insurance Claim Prediction System")

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter Vehicle & Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    subscription_length = st.number_input(
        "Subscription Length",
        min_value=0.0,
        value=11.8
    )

    vehicle_age = st.number_input(
        "Vehicle Age",
        min_value=0.0,
        value=1.0
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=47
    )

    region_code = st.text_input(
        "Region Code",
        value="C2"
    )

    region_density = st.number_input(
        "Region Density",
        min_value=0,
        value=27003
    )

    segment = st.text_input(
        "Segment",
        value="B2"
    )

    model = st.text_input(
        "Model",
        value="M6"
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    engine_type = st.text_input(
        "Engine Type",
        value="K Series Dual jet"
    )

with col2:

    airbags = st.number_input(
        "Airbags",
        min_value=0,
        value=2
    )

    rear_brakes_type = st.selectbox(
        "Rear Brakes Type",
        ["Drum", "Disc"]
    )

    displacement = st.number_input(
        "Displacement",
        min_value=0,
        value=1197
    )

    cylinder = st.number_input(
        "Cylinder",
        min_value=1,
        value=4
    )

    transmission_type = st.selectbox(
        "Transmission Type",
        ["Manual", "Automatic"]
    )

    steering_type = st.selectbox(
        "Steering Type",
        ["Electric", "Manual", "Power"]
    )

    turning_radius = st.number_input(
        "Turning Radius",
        min_value=0.0,
        value=4.8
    )

    length = st.number_input(
        "Length",
        min_value=0,
        value=3845
    )

    width = st.number_input(
        "Width",
        min_value=0,
        value=1735
    )

    gross_weight = st.number_input(
        "Gross Weight",
        min_value=0,
        value=1335
    )

with col3:

    ncap_rating = st.number_input(
        "NCAP Rating",
        min_value=0,
        value=2
    )

    max_power_bhp = st.number_input(
        "Max Power (BHP)",
        min_value=0.0,
        value=88.5
    )

    max_power_rpm = st.number_input(
        "Max Power RPM",
        min_value=0.0,
        value=6000.0
    )

    max_torque_nm = st.number_input(
        "Max Torque (Nm)",
        min_value=0.0,
        value=113.0
    )

    max_torque_rpm = st.number_input(
        "Max Torque RPM",
        min_value=0.0,
        value=4400.0
    )

# --------------------------------------------------
# BINARY FEATURES
# --------------------------------------------------

st.subheader("Vehicle Safety & Feature Configuration")

binary_columns = [
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert"
]

binary_values = {}

cols = st.columns(3)

for i, feature in enumerate(binary_columns):
    with cols[i % 3]:
        binary_values[feature] = st.selectbox(
            feature.replace("is_", "").replace("_", " ").title(),
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Claim", use_container_width=True):

    payload = {
        "subscription_length": subscription_length,
        "vehicle_age": vehicle_age,
        "customer_age": customer_age,
        "region_code": region_code,
        "region_density": region_density,
        "segment": segment,
        "model": model,
        "fuel_type": fuel_type,
        "engine_type": engine_type,
        "airbags": airbags,

        "is_esc": binary_values["is_esc"],
        "is_adjustable_steering": binary_values["is_adjustable_steering"],
        "is_tpms": binary_values["is_tpms"],
        "is_parking_sensors": binary_values["is_parking_sensors"],
        "is_parking_camera": binary_values["is_parking_camera"],

        "rear_brakes_type": rear_brakes_type,
        "displacement": displacement,
        "cylinder": cylinder,
        "transmission_type": transmission_type,
        "steering_type": steering_type,
        "turning_radius": turning_radius,
        "length": length,
        "width": width,
        "gross_weight": gross_weight,

        "is_front_fog_lights": binary_values["is_front_fog_lights"],
        "is_rear_window_wiper": binary_values["is_rear_window_wiper"],
        "is_rear_window_washer": binary_values["is_rear_window_washer"],
        "is_rear_window_defogger": binary_values["is_rear_window_defogger"],
        "is_brake_assist": binary_values["is_brake_assist"],
        "is_power_door_locks": binary_values["is_power_door_locks"],
        "is_central_locking": binary_values["is_central_locking"],
        "is_power_steering": binary_values["is_power_steering"],
        "is_driver_seat_height_adjustable": binary_values["is_driver_seat_height_adjustable"],
        "is_day_night_rear_view_mirror": binary_values["is_day_night_rear_view_mirror"],
        "is_ecw": binary_values["is_ecw"],
        "is_speed_alert": binary_values["is_speed_alert"],

        "ncap_rating": ncap_rating,
        "max_power_bhp": max_power_bhp,
        "max_power_rpm": max_power_rpm,
        "max_torque_nm": max_torque_nm,
        "max_torque_rpm": max_torque_rpm
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["claim_prediction"]
            probability = result["claim_probability"]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Claim Likely")
            else:
                st.success("✅ Claim Unlikely")

            st.metric(
                "Claim Probability",
                f"{probability * 100:.2f}%"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.json(response.json())

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI. "
            "Make sure the API is running on port 8000."
        )