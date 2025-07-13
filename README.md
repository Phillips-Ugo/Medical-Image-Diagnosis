# 🧠 AI-Powered Skin Disease Diagnosis with GAN Augmentation

This project is an AI-based healthcare application designed to assist in the early detection of common skin diseases. It leverages Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs) to augment imbalanced medical datasets and improve diagnostic performance. A Streamlit-powered web app serves as the user interface, where users can upload skin images, get diagnosis predictions, and interact with a **medical chatbot**   for treatment guidance.

---

## 💡 Key Features

- ✅ **Skin Disease Classification** using a trained CNN (EfficientNet)
- ⚖️ **Class Imbalance Mitigation** via GAN-based synthetic image generation
- 🧬 **Medical Chatbot** providing condition overviews, symptoms, and treatment options
- 📍 **Nearby Medical Facility Finder** based on geolocation
- 📊 **Diagnosis History** saved locally or in a database
- 🌐 **Streamlit Web App** for intuitive and accessible interaction

---

## 🚀 How It Works

### 1. Dataset Preprocessing
- Uses the HAM10000 for skin disease images.
- Images are resized, normalized, and split into training and validation sets.

### 2. Handling Class Imbalance
- Trained a **DCGAN** (Deep Convolutional GAN) to generate synthetic images for underrepresented classes.
- Augmented dataset is used to train the CNN model, improving accuracy on rare conditions like Melanoma.

### 3. Model Architecture
- Feature extractor: `EfficientNetB0` fine-tuned on the augmented dataset.
- Trained using TensorFlow and Keras, with callbacks and early stopping.

### 4. Streamlit Web App
- Upload a skin image → classify using the trained CNN → show predicted condition.
- Chatbot answers questions and provides treatment advice.
- Optional: Locate nearby hospitals using geopy and OpenStreetMap.

---

## 💬 Chatbot Features

- Powered by a fine-tuned GPT-based assistant.
- Trained on condition-specific examples (e.g., symptoms, causes, treatments).
- Capable of answering questions like:
  - “What are the symptoms of Eczema?”
  - “How can I treat Melanoma?”
  - “Is this condition contagious?”

---

## 🛠️ Installation

### 🔧 Requirements
- Python >= 3.8
- TensorFlow
- Keras
- NumPy, Pandas, Matplotlib
- Streamlit
- PIL (Pillow)
- geopy
- OpenAI API (for chatbot)

### ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-skin-diagnosis-app.git
cd ai-skin-diagnosis-app

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/app.py

🧪 Example Use
Launch the app.
Upload an image of a skin lesion.
View the diagnosis and probability scores.
Ask the chatbot for treatment or more info.
See a map of nearby clinics if location is enabled.

📈 Results
GAN-augmented model accuracy: 90% (vs baseline: 82%)
F1-score improvement on minority classes: +10%


