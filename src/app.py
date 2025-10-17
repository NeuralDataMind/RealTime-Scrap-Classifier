import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import collections
import queue
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real-Time Scrap Classifier",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Real-Time Scrap Classifier Dashboard")
st.write("This dashboard uses a YOLOv8 model to detect and classify scrap items in real-time.")

# --- MODEL LOADING ---
MODEL_PATH = 'Best.pt' 

@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from path '{path}': {e}")
        return None

model = load_yolo_model(MODEL_PATH)

# This queue will safely pass detection results from the video thread to the main thread
result_queue: "queue.Queue[list[str]]" = queue.Queue()

# --- REAL-TIME DETECTION LOGIC ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    if model is None:
        return frame

    img = frame.to_ndarray(format="bgr24")
    img_resized = cv2.resize(img, (640, 480))

    results = model(img_resized, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    
    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
    if detected_classes:
        result_queue.put(detected_classes)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- STREAMLIT LAYOUT ---
# Main section for the live dashboard
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Live Conveyor Simulation")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="classifier",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.header("Detection Statistics")
    st.write("Counts will update here as objects are detected.")
    
    if 'counts' not in st.session_state:
        st.session_state.counts = collections.Counter()

    stats_placeholder = st.empty()

# Continuously update the stats from the queue
while ctx.state.playing:
    try:
        detected_classes = result_queue.get(timeout=1)
        st.session_state.counts.update(detected_classes)
    except queue.Empty:
        pass

    with stats_placeholder.container():
        st.write("---")
        st.write("**Total Items Detected:**")
        
        if not st.session_state.counts:
            st.info("No items detected yet.")
        else:
            for class_name, count in st.session_state.counts.items():
                st.markdown(f"- **{class_name}:** `{count}`")
        st.write("---")


# --- SIDEBAR FOR IMAGE UPLOAD ---
st.sidebar.title("Test with a Single Image")
uploaded_image = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and model is not None:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Run detection on the uploaded image
    st.sidebar.image(image, channels="BGR", caption="Uploaded Image")
    
    results = model(image, conf=0.4, verbose=False)
    annotated_image = results[0].plot()

    st.sidebar.image(annotated_image, channels="BGR", caption="Detection Result")
    
    # Display detected object classes in the sidebar
    detected_items = [model.names[int(c)] for c in results[0].boxes.cls]
    if detected_items:
        st.sidebar.success(f"Detected: {', '.join(detected_items)}")
    else:
        st.sidebar.warning("No objects detected in the image.")

elif model is None:
    st.sidebar.error("Model is not loaded. Cannot process image.")