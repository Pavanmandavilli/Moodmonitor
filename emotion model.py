import cv2
from deepface import DeepFace

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use DeepFace to analyze emotions
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Check if results are in a list format
    if isinstance(results, list):
        for result in results:
            # Extract dominant emotion
            dominant_emotion = result['dominant_emotion']
            # Get the coordinates for the face bounding box
            face_rect = result['region']
            x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # If not a list, handle single detection
        dominant_emotion = results['dominant_emotion']
        face_rect = results['region']
        x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
