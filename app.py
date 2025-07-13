# AI Medical Diagnosis App with Chat Bot
# Developed by Ogwumike Ugochukwu Belusochim

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import streamlit as st
import numpy as np
from PIL import Image
import time
import datetime
import os
import pickle
import pandas as pd
import tensorflow as tf
import requests
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv() # This loads the variables from .env into the environment

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Chat Bot Functions ---
def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your medical assistant. How can I help you today?"}
        ]

# Load API keys from environment variables

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_ai_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available and needed
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable medical assistant."},
                {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error. Please try again."

# --- Get Coordinates Function ---
def get_coordinates(address):
    geolocator = Nominatim(user_agent="med_app")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

# --- Get Nearby Hospitals using Google Places API ---
def get_nearby_hospitals(lat, lon, radius=5000):
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lon}&radius={radius}&type=hospital&key={GOOGLE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    hospitals = []  
    if data.get("results"):
        for place in data["results"][:5]:  # Limit to closest 5 hospitals
            name = place.get("name")
            location = place["geometry"]["location"]
            place_lat = location["lat"]
            place_lon = location["lng"]
            distance_km = geodesic((lat, lon), (place_lat, place_lon)).km
            distance_miles = distance_km * 0.621371
            address = place.get("vicinity", "N/A")
            directions_url = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={place_lat},{place_lon}&travelmode=driving"

            hospitals.append({
                "name": name,
                "distance_km": round(distance_km, 2),
                "distance_miles": round(distance_miles, 2),
                "address": address,
                "directions_url": directions_url
            })
    return hospitals

# --- Save Diagnosis History ---
def save_history(image_name, diagnosis, confidence):
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "image": image_name,
        "diagnosis": diagnosis,
        "confidence": f"{confidence*100:.2f}%",
    }

    if os.path.exists("diagnosis_history.csv"):
        df = pd.read_csv("diagnosis_history.csv")
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:   
        df = pd.DataFrame([record])
    
    df.to_csv("diagnosis_history.csv", index=False)

# --- Load pre-trained model ---
@st.cache_resource
def load_model():
    return pickle.load(open("HealthCareModel.h5", 'rb'))

model = load_model()

# --- Preprocessing function ---
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    image_array = img_to_array(img)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- Prediction function ---
def predict_disease(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# --- Disease Info Knowledge Base ---
def get_disease_info(diagnosis):
    info = {
        "Benign keratosis-like lesions": {
            "Overview": "These are non-cancerous (benign) skin growths that often appear with age or from long-term sun exposure. They are usually harmless and don't require treatment unless they become bothersome.",
            "Symptoms": "Flat or slightly raised patches that can feel rough, scaly, or crusty. They may be tan, brown, black, or pink and often look similar to warts or moles.",
            "Tips": "Use gentle skin moisturizers, avoid scratching or picking at the lesion, and protect your skin from sun exposure with SPF 30+ sunscreen.",
            "When to see a doctor": "If the lesion changes in color, shape, or size, becomes painful, starts bleeding, or looks significantly different from others.",
            "Treatment": "Usually not needed. If necessary, treatments include cryotherapy (freezing), curettage (scraping), or laser therapy.",
            "Learn More": "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"  # Mayo Clinic source
        },
        "Basal cell carcinoma": {
            "Overview": "A common and slow-growing form of skin cancer caused mainly by chronic sun damage. It rarely spreads but can damage surrounding tissue if untreated.",
            "Symptoms": "A shiny, pearly bump or a flat, scaly patch. It may look like a sore that won't heal or bleeds easily.",
            "Tips": "Apply broad-spectrum sunscreen daily, avoid tanning beds, wear protective clothing, and have your skin checked regularly.",
            "When to see a doctor": "If you notice a persistent sore, a new growth, or a lesion that bleeds, scabs, and doesn't heal.",
            "Treatment": "Early detection is key. Options include surgical removal, topical treatments, cryotherapy, or radiation therapy depending on severity.",
            "Learn More": "https://www.skincancer.org/skin-cancer-information/basal-cell-carcinoma/"  # Skin Cancer Foundation
        },
        "Actinic keratoses": {
            "Overview": "Precancerous patches of thick, scaly skin that form from years of sun exposure. They can turn into squamous cell carcinoma if not treated.",
            "Symptoms": "Dry, rough, or crusty patches that are usually pink, red, or skin-colored, often appearing on the face, ears, neck, or hands.",
            "Tips": "Use sunscreen daily, avoid prolonged sun exposure (especially midday), and wear a wide-brimmed hat and protective clothing.",
            "When to see a doctor": "If the patch becomes red, tender, starts bleeding, or grows larger.",
            "Treatment": "May be treated with cryotherapy, topical creams (like fluorouracil), chemical peels, or light-based therapies (photodynamic therapy).",
            "Learn More": "https://www.skincancer.org/skin-cancer-information/actinic-keratosis/"  # Skin Cancer Foundation
        },
        "Vascular lesions": {
            "Overview": "These are skin markings caused by abnormal blood vessels, such as hemangiomas, port-wine stains, or spider veins. Most are harmless, but some may need treatment.",
            "Symptoms": "Flat or raised red, blue, or purple spots or patches on the skin. Some lesions may darken or become raised over time.",
            "Tips": "Avoid trauma to the area, especially in infants and children. Camouflage makeup can be used for cosmetic reasons.",
            "When to see a doctor": "If the lesion grows rapidly, bleeds easily, causes pain, or affects vision or breathing (in rare cases).",
            "Treatment": "Options include laser therapy, sclerotherapy (for spider veins), or surgical removal if necessary.",
            "Learn More": "https://www.mayoclinic.org/diseases-conditions/hemangioma/symptoms-causes/syc-20352334"  # Mayo Clinic for hemangiomas
        },
        "Melanocytic nevi": {
            "Overview": "Commonly known as moles, these are small collections of pigment-producing cells. Most are benign and appear during childhood or adolescence.",
            "Symptoms": "Usually round or oval spots that can be flat or raised, with smooth edges. They range in color from light brown to black.",
            "Tips": "Examine your skin monthly using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution) to spot unusual moles.",
            "When to see a doctor": "If a mole itches, bleeds, or changes in size, color, or shape, especially if it's new or looks different from others (a so-called 'ugly duckling').",
            "Treatment": "Most don't require removal. Suspicious or cosmetically unwanted moles can be biopsied or removed surgically.",
            "Learn More": "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"  # Mayo Clinic source
        },
        "Melanoma": {
            "Overview": "A dangerous and fast-spreading form of skin cancer that arises from pigment cells (melanocytes). Early detection can be life-saving.",
            "Symptoms": "A new or changing mole that may be asymmetric, have uneven borders, multiple colors, be larger than 6mm, or evolving over time.",
            "Tips": "Never ignore changes in skin spots. Avoid sunburns, use broad-spectrum SPF 50+, and get annual full-body skin exams.",
            "When to see a doctor": "Immediately if any mole meets ABCDE criteria or looks noticeably different from others.",
            "Treatment": "Usually requires surgical removal. Advanced cases may need immunotherapy, targeted therapy, chemotherapy, or radiation.",
            "Learn More": "https://www.skincancer.org/skin-cancer-information/melanoma/"  # Skin Cancer Foundation
        },
        "Dermatofibroma": {
            "Overview": "A harmless skin nodule that often forms after minor skin injury, insect bite, or shaving. It's made of fibrous tissue and doesn't pose a health risk.",
            "Symptoms": "Firm, dome-shaped bump that may feel like a hard lump under the skin. It's usually brownish or reddish and can dimple inward when pinched.",
            "Tips": "Avoid shaving over the bump, don't squeeze or scratch it. It typically doesn't grow or change much.",
            "When to see a doctor": "If becomes painful, bleeds, grows rapidly, or if you're uncertain about the diagnosis.",
            "Treatment": "Generally no treatment needed. If it's bothersome, it can be removed surgically, though this may leave a scar.",
            "Learn More": "https://dermnetnz.org/topics/dermatofibroma"  # DermNet NZ (reputable dermatology resource)
        }
    }

    return info.get(diagnosis, {
        "Overview": "Information not available.",
        "Symptoms": "N/A",
        "Tips": "N/A",
        "When to see a doctor": "N/A",
        "Treatment": "N/A",
        "Learn More": None
    })

# --- Streamlit App Config ---
st.set_page_config(page_title="AI Medical Diagnosis", page_icon="🩺", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .title {
            font-size:48px;
            font-weight: 800;
            color: #2c3e50;
        }
        .subtitle {
            font-size:20px;
            color: #34495e;
        }
        .stButton>button {
            background-color: #2980b9;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #1abc9c;
        }
        /* Chat bot styles */
        .chat-container {
            width: 100%;
            height: 500px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background-color: #D3D3D3;
            border-bottom: 1px solid #ccc;  
            color: white;
            padding: 0px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-messages {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .chat-input {
            padding: 15px;
            border-top: 1px solid #eee;
            background-color: white;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        .assistant-message {
            background-color: #f1f1f1;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        .chat-page {
            max-width: 800px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chat
initialize_chat()

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
selection = st.sidebar.radio("Go to", ["🔍 Diagnose", "📂 History", "💬 AI Assistant", "🏥 Nearby Hospitals", "ℹ️ About"])

# --- Main Content ---
if selection == "🔍 Diagnose":
    st.markdown('<p class="title">AI Medical Image Diagnosis 🧠</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a medical image for instant insights</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg"])
    col1, col2 = st.columns([1, 2])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction & Insights")
            run_diagnosis = st.button("🧪 Run Diagnosis", disabled=(uploaded_file is None))
            if run_diagnosis:
                with st.spinner("Analyzing..."):
                    try:
                        predicted_class, confidence = predict_disease(image)
                        diagnosis_map = {
                            0: "Benign keratosis-like lesions",
                            1: "Basal cell carcinoma",
                            2: "Actinic keratoses",
                            3: "Vascular lesions",
                            4: "Melanocytic nevi",
                            5: "Melanoma",
                            6: "Dermatofibroma"
                        }
                        result = diagnosis_map.get(predicted_class, "Unknown")
                        save_history(uploaded_file.name, result, confidence)

                        st.success("Diagnosis Complete!")
                        st.markdown(f"### 🧾 Result: **{result}**")
                        st.markdown(f"**Confidence Score:** {confidence*100:.2f}%")

                        disease_info = get_disease_info(result)
                        with st.expander("📘 Learn More About This Condition"):
                            st.subheader("Overview")
                            st.write(disease_info["Overview"])

                            st.subheader("Common Symptoms")
                            st.write(disease_info["Symptoms"])

                            st.subheader("Self-Care Tips")
                            st.write(disease_info["Tips"])

                            st.subheader("When to See a Doctor")
                            st.write(disease_info["When to see a doctor"])

                            st.subheader("Possible Treatments")
                            st.write(disease_info["Treatment"])
                    except Exception as e:
                        st.error(f"Error during diagnosis: {str(e)}")
    else:
        with col2:
            st.subheader("Prediction & Insights")
            st.button("🧪 Run Diagnosis", disabled=True)

# --- History Section ---
elif selection == "📂 History":
    st.markdown('<p class="title">🗂️ Diagnosis History</p>', unsafe_allow_html=True)
    if os.path.exists("diagnosis_history.csv"):
        df = pd.read_csv("diagnosis_history.csv")
        st.dataframe(df[df.shape[0]-5:]) # Get last 5 records
    else:
        st.info("No diagnosis history yet.")

# --- AI Assistant (Chat Bot) Section ---
elif selection == "💬 AI Assistant":
    st.markdown("""
    <style>
        .chat-container {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(128, 128, 128, 0.5);
            padding: 0;
            display: flex;
            flex-direction: column;
        }
        .chat-input-container {
            padding: 15px;
            border-top: 1px solid #444444; /* subtle border */
            border-radius: 0 0 10px 10px;
            color: #eee; /* text color for input area */
        }
        .chat-messages {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;   
            color: #000; /* message text color */
        }
        .chat-input-container {
            padding: 15px;
            border-radius: 0 0 10px 10px;
        }
        .user-message {
            background-color: #444444;
            border-radius: 18px 18px 0 18px;
            padding: 10px 16px;
            margin-left: auto;
            margin-bottom: 8px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .assistant-message {
            background-color: #444444;
            border-radius: 18px 18px 18px 0;
            padding: 10px 16px;
            margin-right: auto;
            margin-bottom: 8px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .message-time {
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">💬 Medical AI Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask me anything about medical conditions, symptoms, or treatments</p>', unsafe_allow_html=True)
    # Chat header
    st.markdown('<div class="chat-header">Medical Assistant Chat</div>', unsafe_allow_html=True)

    # Chat messages area
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                    <div class="user-message">
                        {message["content"]}
                        <div class="message-time">{datetime.datetime.now().strftime("%H:%M")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                    <div class="assistant-message">
                        {message["content"]}
                        <div class="message-time">{datetime.datetime.now().strftime("%H:%M")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # Chat input area
    if prompt := st.chat_input("Type your medical question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.spinner("Thinking..."):
            response = generate_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Rerun to update the chat display
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-container
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
# --- Nearby Hospital Section ----
elif selection == "🏥 Nearby Hospitals":
    st.markdown('<p class="title">Find Nearby Hospitals 🏥</p>', unsafe_allow_html=True)
    st.write("Enter your location to find nearby hospitals.")
    address = st.text_input("Enter your address or location:")
    if address:
        try:
            lat, lon = get_coordinates(address)
            if lat and lon:
                hospitals = get_nearby_hospitals(lat, lon)
                if hospitals:
                    sorted_hospitals = sorted(hospitals, key=lambda x: x['distance_miles'])
                    st.success(f"Found {len(hospitals)} hospitals near you:")
                    for hospital in sorted_hospitals:
                        st.markdown(
                            f"**{hospital['name']}**  \n📍 {hospital['address']}  \n"
                            f"🚗 {hospital['distance_miles']} miles away  \n"
                            f"[🗺️ View on Google Maps]({hospital['directions_url']})"
                        )
                else:
                    st.warning("No hospitals found nearby.")
            else:
                st.error("Could not find your location. Please try a more specific address.")
        except Exception as e:
            st.error(f"Error retrieving hospital data: {str(e)}")

# --- About Section ---
elif selection == "ℹ️ About":
    st.markdown('<p class="title">About This App 📘</p>', unsafe_allow_html=True)
    st.write("""
        This AI-powered app helps diagnose skin and medical conditions for Seven Different Classes: 
        Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions, Dermatofibroma, 
        Melanoma, Melanocytic nevi, and Vascular lesions.
        \n
        Upload an image and get instant analysis powered by a pre-trained EfficientNet model.
        The AI assistant chat bot is here to answer your medical questions and guide you through.
        \n
        Developed by Ogwumike Ugochukwu Belusochim.
    """)

