import cv2
import speech_recognition as sr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from deepface import DeepFace

cap = cv2.VideoCapture(0)

recognizer = sr.Recognizer()

model_name = 'j-hartmann/emotion-english-distilroberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token="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")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")

emotion_analysis = pipeline('text-classification', model=model, tokenizer=tokenizer)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    if isinstance(results, list):
        for result in results:
            dominant_emotion = result['dominant_emotion']
            face_rect = result['region']
            x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Facial Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        dominant_emotion = results['dominant_emotion']
        face_rect = results['region']
        x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Facial Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Facial Emotion Detection', frame)

    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")

            emotions = emotion_analysis(text)
            for emotion in emotions:
                label = emotion['label']
                score = emotion['score']
                print(f"Audio Emotion: {label}, Confidence: {score:.2f}")

        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
