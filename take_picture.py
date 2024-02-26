import cv2
import os

def start():
    # Directory to save the dataset
    dataset_directory = 'dataset'

    # Create the directory if it doesn't exist
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    # Initialize the webcam or video capture device
    video_capture = cv2.VideoCapture(0)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Prompt the user to input the label name
        print("press q twice then again type q in input to quit camera.")
        label_name = input("Enter person's  name (or 'q' to quit): ")

        if label_name.lower() == 'q':
            break

        # Create the label directory if it doesn't exist
        label_directory = os.path.join(dataset_directory, label_name)
        if not os.path.exists(label_directory):
            os.makedirs(label_directory)

        # Counter for the images
        image_counter = 0

        # Loop to capture images
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterate over each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Crop the face region from the frame
                face_roi = frame[y:y+h, x:x+w]

                # Save the cropped face image to the label directory
                face_image_path = os.path.join(label_directory, 'face_{}_{}.jpg'.format(label_name, image_counter))
                cv2.imwrite(face_image_path, face_roi)
                print('Face {} saved to {}'.format(image_counter, label_name))

                # Increment the image counter
                image_counter += 1

                # Break the loop if 200 photos are taken
                if image_counter == 200:
                    break

            # Display the frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' is pressed or 200 photos are taken
            if cv2.waitKey(1) & 0xFF == ord('q') or image_counter == 200:
                break

        # Break the loop if 'q' is pressed or 200 photos are taken
        if cv2.waitKey(1) & 0xFF == ord('q') or image_counter == 200:
            break

    # Release the video capture device and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


