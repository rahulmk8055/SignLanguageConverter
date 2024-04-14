import json
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from state import State
from time import time
from streamlit_pills import pills

state = State()
state.load_state()

confidence = 0


def write_value_to_file(value):
    with open("gauge_value.json", "w") as f:
        json.dump({"value": value}, f)


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

labels_dict = {0: 'M', 1: 'N', 2: 'P', 3: 'Q', 4: 'R'}
detection_start_time = None
last_detected_character = ""
hold_time = 2  # seconds

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

st.markdown("<h2 style='text-align: center;'>ASL Interpretation Tool</h1>", unsafe_allow_html=True)
st.markdown('#')

value = ""


def clear_word():
    state.set_current_word("")


with st.container():
    col1, col2 = st.columns([0.6, 0.4], gap='large')
    with st.container(border=True):
        with col1:
            videoStream = st.empty()
            col3, col4 = st.columns([0.6, 0.4])
            with col3:
                value = pills("Select Letter For Sign Reference",
                              ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        with col4:
            st.image("B.png", width=250)

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
            '<iframe src="http://localhost:8001/gauge_graph.html" width="800" height="240"></iframe>',
            unsafe_allow_html=True
        )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        x_, y_, data_aux = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        probabilities = model.predict_proba([np.asarray(data_aux)])
        confidence = max(max(probabilities))

        predicted_character = labels_dict[int(prediction[0])]
        # predictedLetterDisplay.header("Predicted Letter: {letter}".format(letter=predicted_character))
        # Check if detected character is consistent
        if predicted_character == last_detected_character:
            if detection_start_time is None:
                detection_start_time = time()
            elif time() - detection_start_time >= hold_time:
                state.set_predicted_letter(predicted_character)
                state.append_to_current_word(predicted_character)
                detection_start_time = None  # Reset the timer
        else:
            last_detected_character = predicted_character
            detection_start_time = None  # Reset the timer
    else:
        last_detected_character = ""
        detection_start_time = None  # Reset the timer if no hands are detected

    progress_value = 0
    if detection_start_time:
        elapsed_time = time() - detection_start_time
        progress_value = int((elapsed_time / hold_time) * 100)
        if elapsed_time < hold_time:
            progress_bar.progress(progress_value)
            predictedLetterDisplay.header("Predicted Letter: {letter}".format(letter=predicted_character))

        else:
            progress_bar.progress(100)
            predictedLetterDisplay.header("Predicted Letter: {letter}".format(letter=predicted_character))
    else:
        progress_bar.progress(0)  # Clears the progress bar if conditions aren't met
        predictedLetterDisplay.header("Predicted Letter: ")

    wordDisplay.header("Predicted Word: :rainbow[{word}]".format(word=state.get_current_word()))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 360))
    videoStream.image(frame, channels="RGB", use_column_width="auto")

    write_value_to_file(confidence * 100)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
