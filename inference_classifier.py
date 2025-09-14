import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands and drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'What', 1: 'My name is', 2: 'Eat', 3: 'No', 4: 'I love you', 5: 'Help', 6: 'Stop', 7: 'Where', 8: 'Drink'}
while True:
    data_aux = []  # Start with an empty list
    x_ = []
    y_ = []
    
    ret, frame = cap.read()

    H, W, _ = frame.shape

    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extract landmarks (x, y) and append them to data_aux
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

    # Ensure data_aux is a NumPy array and has the correct shape before padding
    if data_aux:  # Check if data_aux is not empty
        data_aux = np.array(data_aux)

        # If data_aux is not the required length (219), pad it
        if data_aux.shape[0] < 219:
            data_aux = np.pad(data_aux, (0, 219 - data_aux.shape[0]), mode='constant', constant_values=0)

        # Make a prediction with the model
        prediction = model.predict([data_aux])  # Make sure the input is in the right format

        # Optional: Display prediction result or processed frame
        # print("Prediction:", prediction)
        predicted_character = labels_dict[int(prediction[0])]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

