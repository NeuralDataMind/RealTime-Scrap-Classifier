# Real-Time Scrap Classifier & Robotic Pick Simulation

**Submission for the Computer Vision Engineer Intern Assignment.**

This project simulates an industrial AI vision pipeline for a scrap sorting system. It uses a custom-trained YOLOv8 model to perform real-time detection and classification of various scrap materials from a live video stream. The final application is a web-based dashboard that displays the live feed, overlays detections, and shows real-time statistics.

---

### Demo

*[Insert a link to your 1-2 minute demo video here. You can upload it to YouTube (unlisted) or Google Drive and paste the shareable link.]*

---

### How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [Your GitHub Repository URL]
    cd [Your-Repo-Name]
    ```

2.  **Install Dependencies:**
    It is recommended to use a Python virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard:**
    This command will launch the Streamlit web application.
    ```bash
    streamlit run src/app.py
    ```
    Your web browser should open a new tab with the dashboard. If not, navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

### My Approach

My development process followed these key steps:

1.  **Data Exploration & Analysis:** I began by researching public datasets for garbage and scrap detection. My initial analysis of a large dataset revealed a **severe class imbalance**, with over 70% of instances belonging to a single category. This would have resulted in a heavily biased and ineffective model.

2.  **Strategic Data Curation:** Recognizing that model performance depends heavily on data quality, I made the decision to switch to a different dataset ("Garbage Detection – 6 Waste Categories" from Kaggle). This dataset, while still moderately imbalanced, provided a sufficient number of images for multiple key classes, creating a much better foundation for training.

3.  **Model Training:** I fine-tuned a lightweight **YOLOv8n** model on the selected dataset. Training was performed in Google Colab to leverage GPU acceleration, which significantly reduced the time required and allowed for more training epochs.

4.  **Real-Time Simulation & Dashboard:** I developed the final application using **Streamlit**. The dashboard provides a user-friendly interface to view the live webcam feed, see the model's detections in real-time, and monitor live statistics on the types and counts of materials detected. It also includes a feature for testing the model on single uploaded images.

---

### Challenges Faced & Key Decisions

* **Challenge: Severe Class Imbalance:** The most significant challenge was the poor quality and severe class imbalance found in many public garbage datasets. My first attempt with a large, imbalanced dataset would have produced a model that was an expert at detecting one class and completely ignorant of others.

* **Key Decision: Prioritizing Data Quality Over Quantity:** Instead of trying to force a flawed dataset to work, my key decision was to **switch to a different dataset** that was better balanced and more suitable for the task, even if it meant restarting the data preparation process. This decision was crucial for the final success of the model. This highlights the real-world principle that the quality of training data is often more important than the sheer quantity.

* **Challenge: Real-time Performance:** The initial version of the Streamlit dashboard froze frequently because the YOLOv8 model was too resource-intensive for the live video thread. I resolved this by resizing the video frames before processing and implementing a thread-safe queue to decouple the video rendering from the model inference, resulting in a smooth user experience.

---

### Libraries Used

* `ultralytics` (for YOLOv8)
* `streamlit`
* `opencv-python`
* `streamlit-webrtc`
* `av`
* `numpy`
