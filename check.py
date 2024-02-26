import cv2
from joblib import load

def start():
    # Load the trained model from disk
    model_file = 'face_recognition_model.joblib'
    clf = load(model_file)

    # Initialize the webcam or video capture device
    video_capture = cv2.VideoCapture(0)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_roi = gray[y:y+h, x:x+w]

            # Resize the face region to match the input size of the model
            face_roi_resized = cv2.resize(face_roi, (160, 160))

            # Flatten the face region
            face_features = face_roi_resized.flatten()

            # Predict the label (name) of the person
            predicted_label = clf.predict([face_features])[0]

            # Draw a rectangle around the detected face and display the predicted label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


