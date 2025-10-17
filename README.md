### **1. Final Project Structure**
# Real-Time Scrap Classifier & Robotic Pick Simulation

**Submission for the Computer Vision Engineer Intern Assignment.**

This project simulates an industrial AI vision pipeline for a scrap sorting system. It uses a custom-trained YOLOv8 model to perform real-time detection of various scrap materials. The final application is an interactive web dashboard built with Streamlit that provides a live webcam feed for detection and a feature to test single images.

---

### Demo

video is provided in the repo as demo.mp4

---

### Features

* **Live Detection:** Uses a webcam to simulate a conveyor belt, performing real-time object detection and classification.
* **Interactive Dashboard:** A user-friendly web interface that displays the live video, detection bounding boxes, and real-time statistics on detected items.
* **Image Upload:** A sidebar option to upload a single image (`.jpg`, `.png`) to test the model's performance on static files. Sample images are provided in the `assets` folder.
* It can detect Plastic, metal, Biodegradable, cardboard, glass, paper

---

### Project Structure

```

RealTime-Scrap-Classifier/
│
├── assets/             \# Contains sample images for testing
│   ├── test\_can.jpg
│   └── test\_bottle.jpg
│
├── src/
│   └── app.py          \# The main Streamlit application script
│
├── Best.pt             \# The trained YOLOv8 model weights
├── README.md           \# Project documentation
└── requirements.txt    \# Required Python libraries

````

---

### Setup and Installation

Follow these steps to set up and run the project locally.

#### **Prerequisites**
* Python 3.8 or higher.
* `pip` for package management.

#### **1. Clone the Repository**
```bash
git clone [https://github.com/NeuralDataMind/RealTime-Scrap-Classifier]
cd [RealTime-Scrap-Classifier]
````

#### **2. Set Up a Virtual Environment (Recommended)**

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **4. Place the Model File**

Ensure that the trained model file, `Best.pt`, is placed in the root directory of the project, at the same level as the `src` folder.

-----

### How to Run

Once the setup is complete, launch the Streamlit dashboard with this command from the project's root directory:

```bash
streamlit run src/app.py
```

Your web browser should open a new tab with the dashboard. If not, navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

-----

### Troubleshooting

  * **Error: `File does not exist: Best.pt`**

      * **Cause:** The script cannot find the model file.
      * **Solution:** Make sure the `Best.pt` file is in the main project folder, not inside the `src` folder. The `app.py` script uses a relative path to find it.

  * **Webcam Freezes or is Slow**

      * **Cause:** Your computer's CPU is being overloaded by the model.
      * **Solution:** The application already resizes the video frames to reduce the load. Ensure no other resource-intensive programs are running.

-----

### **Note on PDF Write-up**

The following sections ("My Approach" and "Challenges Faced") provide a summary for this README. These topics should be **expanded upon in much greater detail** in your separate PDF submission, as they demonstrate your engineering thought process.

#### My Approach

I began by analyzing public garbage datasets, where I identified a severe class imbalance. Recognizing that data quality is paramount, I switched to a more suitable dataset ("Garbage Detection – 6 Waste Categories") to train a lightweight YOLOv8n model. The final application was built with Streamlit to provide an interactive dashboard for real-time detection and statistical tracking.

#### Challenges Faced & Key Decisions

  * **Challenge:** The primary challenge was the **severe class imbalance** in initial datasets, which would have produced a biased model.
  * **Key Decision:** My most critical decision was to **prioritize data quality over quantity** by switching to a better-balanced dataset. This was essential for the final model's success.
  * **Challenge:** The live dashboard initially froze due to performance bottlenecks. I resolved this by implementing a **thread-safe queue** to decouple video rendering from model inference, ensuring a smooth user experience.

<!-- end list -->

```
```
