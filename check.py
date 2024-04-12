import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from state import State
from time import time

state = State()

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
hold_time = 3  # seconds

col1, col2 = st.columns([0.9, 0.1])
videoStream = col1.empty()
wordStream = col2.empty()

with col2:
    st.title("Predicted Letter")
    predictedLetterDisplay = st.empty()
    wordDisplay = st.empty()
    progress_bar = st.empty()

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
        predicted_character = labels_dict[int(prediction[0])]
        predictedLetterDisplay.text(predicted_character)
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
        else:
            progress_bar.progress(100)
    else:
        progress_bar.empty()  # Clears the progress bar if conditions aren't met

    # predictedLetterDisplay.text(state.get_predicted_letter())
    wordDisplay.text(state.get_current_word())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    videoStream.image(frame, channels="RGB")

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
