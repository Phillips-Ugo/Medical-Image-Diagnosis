    
---

# 🧠 AI-Powered Skin Disease Diagnosis with GAN-Based Dark Skin Image Augmentation

This project is an AI-driven healthcare application aimed at improving early detection of common skin diseases across diverse skin tones. It leverages Convolutional Neural Networks (EfficientNet) combined with Generative Adversarial Networks (GANs) to generate synthetic images of darker skin tones, addressing dataset bias and class imbalance. The result is a more accurate and equitable diagnostic tool accessible through a user-friendly Streamlit web app. Users can upload skin lesion images, receive diagnosis predictions, interact with a medical chatbot for detailed condition information and treatment guidance, and find nearby hospitals/clinics for emergency or personal purposes.

---

## 💡 Key Features

* ✅ **Skin Disease Classification** with a fine-tuned EfficientNet CNN model
* ⚖️ **Class Imbalance Handling** using GAN-generated synthetic dark skin images and resampling techniques
* 🧬 **Medical Chatbot** A fine-tuned LLM trained on possible medical queries that provides condition overviews, symptoms, causes, and treatment options
* 📍 **Nearby Medical Facility Locator** Provides five closest hospitals via geolocation integration and maps API
* 📊 **Diagnosis History Tracking** saved locally
* 🌐 **Streamlit Web Interface** for easy access and interaction

---

## 🚀 How It Works

### 1. Dataset Preprocessing

* Utilizes the HAM10000 and related skin lesion image datasets.
* Images resized, normalized, and divided into training and validation subsets.

### 2. Addressing Class Imbalance & Skin Tone Representation

* Trained **GAN models specifically to generate synthetic images of darker skin tones** to improve representation.
* Applied **resampling methods** (oversampling minority classes and undersampling majority classes) to balance dataset distribution.
* These combined approaches enhance model fairness and diagnostic accuracy across skin tones.

### 3. Model Architecture

* Fine-tuned `EfficientNetB0` CNN on the augmented and balanced dataset.
* Implemented using TensorFlow and Keras, with early stopping and checkpoint callbacks to prevent overfitting.

### 4. Streamlit Web App

* Users upload a skin lesion image → model predicts disease class → results with confidence scores are shown.
* Medical chatbot answers user questions about symptoms, causes, and treatments.
* Location-based hospital/clinic finder is optionally available.

---

## 💬 Medical Chatbot

* Powered by a fine-tuned GPT-based assistant trained on dermatology-specific dialogues.
* Supports queries such as:

  * “What are the symptoms of Melanoma?”
  * “How can I treat Actinic Keratoses?”
  * “Is this condition contagious?”

---

## 📈 Performance Highlights

| Metric                        | Baseline Model | GAN + Resampling Model | Improvement |
| ----------------------------- | -------------- | ---------------------- | ----------- |
| Overall Accuracy              | 82%            | \~90%                  | +8%         |
| F1-Score on Minority Classes  | Lower          | +10%                   | +10%        |
| Fairness on Dark Skin Lesions | Limited        | Significantly Improved | +45%        |

---

## 🛠️ Installation

### Requirements

* Python 3.8 or higher
* TensorFlow
* Keras
* NumPy, Pandas, Matplotlib
* Streamlit
* Pillow (PIL)
* geopy
* OpenAI API (for chatbot)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-skin-diagnosis-app.git
cd ai-skin-diagnosis-app

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

---

## 🧪 Usage

1. Launch the app with `streamlit run app/app.py`.
2. Upload an image of a skin lesion.
3. View the model's diagnosis and confidence scores.
4. Use the chatbot to ask about symptoms, treatments, and more.
5. Optionally, allow location access to find nearby medical facilities.

---

## 🙏 Acknowledgments

* HAM10000 dataset providers
* OpenAI for GPT models powering the chatbot
* TensorFlow and Streamlit communities

---
