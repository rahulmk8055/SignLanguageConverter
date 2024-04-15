import json
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from state import State
from time import time
from streamlit_pills import pills
import imutils
import requests

state = State()
state.load_state()

confidence = 0
predicted_character = ""

url = "http://10.0.0.207:8080/shot.jpg?rnd=679082"
def write_value_to_file(value):
    with open("gauge_value.json", "w") as f:
        json.dump({"value": value}, f)


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://10.0.0.207:8080/h264.sdp')
fps = int(cap.get(cv2.CAP_PROP_FPS))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

detection_start_time = None
last_detected_character = ""
hold_time = 2.5  # seconds

st.set_page_config(page_title="ASL", layout='wide')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Sign Language Interpretation and Learning Tool</h1>",
            unsafe_allow_html=True)
st.markdown('#')

value = "clear"


def clear_word():
    state.set_current_word("")


with st.container():
    col1, col2 = st.columns([0.6, 0.4], gap='large')
    with st.container(border=True):
        with col1:
            videoStream = st.empty()
            col3, col4 = st.columns([0.6, 0.4], gap="large")
            with col3:
                st.header("Select Letter For Sign Reference")
                value = pills("",
                              ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'clear'])
        with col4:
            st.image("./signImages/{value}.png".format(value=value), width=300)

    with col2:
        predictedLetterDisplay = st.empty()
        predictedLetterDisplay.header("Predicted Letter: ")

        progress_bar = st.empty()
        # st.markdown('#')
        st.divider()
        # st.title("Predicted Word")
        wordDisplay = st.empty()
        st.button("clear", on_click=clear_word)
        # st.markdown('#')
        st.divider()
        st.header("Confidence Score")
        st.markdown(
            '<iframe src="http://prompt-major-trout.ngrok-free.app/gauge_graph.html" width="800" height="240"></iframe>',
            unsafe_allow_html=True
        )

while True:

    frame = requests.get(url)
    frame = np.array(bytearray(frame.content), dtype=np.uint8)
    frame = cv2.imdecode(frame, -1)
    # ret, frame = cap.read()
    # if not ret:
    #     break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        x_, y_, data_aux = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            default_style = mp_drawing_styles.get_default_hand_connections_style()

            # Create a new dictionary to store the modified style
            modified_style = {}

            # Iterate through the default style and modify the thickness
            for connection, spec in default_style.items():
                # Create a new DrawingSpec with modified thickness
                modified_spec = DrawingSpec(color=(48, 255, 48), thickness=10)
                # Assign the modified spec to the connection
                modified_style[connection] = modified_spec
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      modified_spec)

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        probabilities = model.predict_proba([np.asarray(data_aux)])
        confidence = max(max(probabilities))
        write_value_to_file(confidence * 100)
        predicted_character = prediction[0]

        state.set_predicted_letter(predicted_character)
        # Check if detected character is consistent
        if predicted_character == last_detected_character:
            if detection_start_time is None:
                detection_start_time = time()
            elif time() - detection_start_time >= hold_time and confidence > 0.8:
                state.append_to_current_word(predicted_character)
                detection_start_time = None  # Reset the timer
        else:
            last_detected_character = predicted_character
            detection_start_time = None  # Reset the timer
    else:
        last_detected_character = ""
        confidence = 0
        write_value_to_file(confidence * 100)
        detection_start_time = None  # Reset the timer if no hands are detected

    progress_value = 0
    colour = "green"
    if confidence < 0.8:
        colour = "red"
    predictedLetterDisplay.header(
        "Predicted Letter: :{colour}[{letter}]".format(letter=predicted_character, colour=colour))
    if detection_start_time:
        elapsed_time = time() - detection_start_time
        progress_value = int((elapsed_time / hold_time) * 100)
        if elapsed_time < hold_time and confidence > 0.8:
            progress_bar.progress(progress_value)
            # predictedLetterDisplay.header("Predicted Letter: {letter}".format(letter=predicted_character))
            # write_value_to_file(confidence * 100)
        elif elapsed_time < hold_time and confidence < 0.8:
            progress_bar.progress(0)
        else:
            progress_bar.progress(100)
    else:
        progress_bar.progress(0)  # Clears the progress bar if conditions aren't met
        predictedLetterDisplay.header("Predicted Letter: ")
        write_value_to_file(0)

    wordDisplay.header("Predicted Word: :green[{word}]".format(word=state.get_current_word()))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 360))
    videoStream.image(frame, channels="RGB", use_column_width="auto")

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
