import cv2
import face_recognition
import os

# Initialize webcam
video_capture = cv2.VideoCapture(0)  # 0 for the default camera

print("Starting real-time face detection. Press 'q' to quit. Press 'n' to select a new user.")

# Initialize user face variables
user_face_encoding = None
user_saved_image_path = "user_face.jpg"
face_recognition_tolerance = 0.45  # Lower tolerance for stricter matching (default is 0.6)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    # Fix mirror image by flipping the frame horizontally
    frame = cv2.flip(frame, 1)  # 1 means horizontal flip

    # Convert the frame from BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use HOG-based model
        if face_locations:
            # Detect face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Compare the detected face encoding to the saved user face encoding
                is_user = (
                    user_face_encoding is not None
                    and face_recognition.compare_faces([user_face_encoding], face_encoding, tolerance=face_recognition_tolerance)[0]
                )
                color = (255, 0, 0) if is_user else (0, 255, 0)  # Blue box for user, green box for others
                label = "User" if is_user else "Other"

                # Draw the box around the face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Add a label below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Save the first detected face as the user if the user has not been selected
                if user_face_encoding is None:
                    user_face_encoding = face_encoding
                    user_image = frame[top:bottom, left:right]  # Crop the user's face
                    cv2.imwrite(user_saved_image_path, user_image)  # Save the face image
                    print("User's face saved as:", user_saved_image_path)

    except Exception as e:
        print("Error during face encoding:", str(e))

    # Display the frame
    cv2.imshow("Real-Time Face Detection", frame)

    # Quit the video by pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        # Reset user face encoding to allow selecting a new user
        user_face_encoding = None
        if os.path.exists(user_saved_image_path):
            os.remove(user_saved_image_path)
        print("User selection reset. The next detected face will be the new user.")

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
