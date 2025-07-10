# 🦿 AI Healthcare: Foot Drop Stimulation Predictor

An AI-powered healthcare assistant designed to help **foot drop** patients by analyzing **x, y, z foot movement data** and predicting the required **electrical stimulation**. The application provides visualization, data analysis, and hospital navigation assistance through an intuitive **Streamlit web interface**.

---

## 🧠 Project Objective

Foot drop is a condition that causes difficulty lifting the front part of the foot. This project assists in:

* ⚡ Predicting the stimulation level needed to aid foot movement
* 📊 Analyzing real-time or uploaded movement data (x, y, z axes)
* 🗺️ Guiding patients to nearby hospitals
* 💵 Supporting the billing process through integrated modules

---

## 🚀 Live Demo

📽️ **Watch the system in action**:
👉 [Click here to view demo](https://your-video-link.com) *(replace with real video/GIF if available)*

---

## 🌟 Key Features

| Feature                            | Description                                                     |
| ---------------------------------- | --------------------------------------------------------------- |
| 🧠 ML-Based Stimulation Prediction | Uses x, y, z foot motion data to predict required stimulation   |
| 📊 Data Visualization              | Line plots, 3D movement graphs, or pressure heatmaps            |
| 📁 Upload or Stream Input          | Accepts CSV files or live data from IMU sensors                 |
| 🏥 Hospital Booking                | Helps users locate and book nearby hospitals                    |
| 💵 Billing Assistance              | Simulates medical billing process for transparency and tracking |
| 📋 Session Logs                    | Maintains logs for patient monitoring and clinician review      |

---

## 🧰 Tech Stack

* **Frontend**: Streamlit
* **ML Models**: Scikit-learn / TensorFlow
* **Visualization**: Plotly, Matplotlib
* **Location Services**: Geopy / Google Maps API *(optional)*
* **Data Input**: CSV Upload or Serial (Arduino IMU)
* **Deployment Ready**: Streamlit Cloud / Heroku compatible

---

## 📦 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Moniica990/AI_Healthcare_Project.git
cd AI_Healthcare_Project

# 2. (Optional) Create virtual environment
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## 📂 Folder Structure

```
📁 AI_Healthcare_Project/
├── 📄 app.py                     # Main Streamlit app
├── 📁 models/                    # ML model files for stimulation prediction
├── 📁 data/                      # Sample foot movement data
│   ├── example_input.csv
├── 📁 utils/                     # Helper functions (prediction, plotting)
├── 📁 pages/                     # Optional: Streamlit multipage views
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧪 Sample Input Format

```csv
timestamp,x,y,z
0.01,0.03,-0.01,9.81
0.02,0.05,-0.02,9.75
...
```

---

## 📌 Use Cases

* 👣 Foot drop rehabilitation for post-stroke or nerve damage patients
* 🏠 Home-based therapy and monitoring
* 🧑‍⚕️ Clinic tool for physiotherapists and neurologists
* 📈 Adaptive feedback system for wearable stimulation devices

---

## 🔮 Future Improvements

* Real-time Bluetooth streaming from Arduino IMU
* Voice-guided assistance for patients
* Role-based dashboard for doctors vs patients
* Cloud data sync & PDF report generation
* Insurance claim integration with billing system

---

📺 Demo
🎥 Watch the App in Action


https://github.com/user-attachments/assets/1b82e56a-d889-471d-95f0-e5fc8d5cad0e



![certificate](https://github.com/user-attachments/assets/04f5b247-e1e9-4832-9bca-49067c0312a1)




