import cv2
import numpy as np
import tensorflow as tf
import time
import requests   # NEW

# Cloud server URL
SERVER_URL = "http://127.0.0.1:5000/predict"

interpreter = tf.lite.Interpreter(model_path="d:/shree/projects/CC ABL/mask_detection_cloud/tflite_model/mask_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# cap = cv2.VideoCapture(0)

# For testing, load an image instead of camera
frame = cv2.imread("d:/shree/projects/CC ABL/mask_detection_cloud/dataset/mask_detection_dataset/data/with_mask/with_mask_1.jpg")
if frame is None:
    print("Failed to load image")
    exit()
print("Image loaded successfully")

# Simulate one frame
ret = True

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"Detected {len(faces)} faces")

for (x, y, w, h) in faces:

    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    face = np.expand_dims(face, axis=0).astype(np.float32)

    start = time.time()

    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    latency = (time.time() - start) * 1000

    label = "Mask" if prediction[0][0] > 0.5 else "No Mask"

    print(f"Edge prediction: {label}, latency: {latency:.2f} ms")

    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.putText(
        frame,
        f"{label} ({latency:.2f} ms)",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    # ---------- SEND DATA TO FRONTEND ----------
    _, img_encoded = cv2.imencode('.jpg', frame)

    files = {
        "image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    data = {
        "edge_prediction": label,
        "edge_latency": str(latency)
    }

    try:
        response = requests.post(SERVER_URL, files=files, data=data)
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print(f"Failed to send, status: {response.status_code}")
    except Exception as e:
        print(f"Failed to send: {e}")
        pass
    # ------------------------------------------

# Since no loop, exit after processing
# cap.release()
cv2.destroyAllWindows()