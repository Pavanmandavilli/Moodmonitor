import speech_recognition as sr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import cv2


recognizer = sr.Recognizer()


model_name = 'j-hartmann/emotion-english-distilroberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token="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")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")


emotion_analysis = pipeline('text-classification', model=model, tokenizer=tokenizer)

while True:
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

        try:

            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")

            emotions = emotion_analysis(text)
            print(f"Emotions: {emotions}")

        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
