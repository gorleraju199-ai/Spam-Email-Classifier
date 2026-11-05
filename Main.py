
# SPAM EMAIL CLASSIFIER
# Email Prediction Program

import pickle
from pathlib import Path

MODEL_PATH = "model/spam_classifier.pkl"

# Check Model
if not Path(MODEL_PATH).exists():
    print("Model not found!")
    print("Please run: python train_model.py")
    raise SystemExit

# Load Model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("\n====================================")
print("       SPAM EMAIL CLASSIFIER")
print("====================================")

print("Enter an email or message to check.")
print("Type 'exit' to close the program.")

while True:

    email = input("\nEnter your message: ").strip()

    if email.lower() == "exit":
        print("Thank you for using our project!")
        break

    if not email:
        print("Please enter a valid message.")
        continue

    # Predict Message
    prediction = model.predict([email])[0]

    probabilities = model.predict_proba([email])[0]

    confidence = max(probabilities) * 100

    print("\n---------- RESULT ----------")

    if prediction == "spam":
        print("WARNING: SPAM MESSAGE DETECTED!")
    else:
        print("MESSAGE CLASSIFIED AS NOT SPAM (HAM).")

    print(f"Prediction Confidence: {confidence:.2f}%")

    print("----------------------------")
  
