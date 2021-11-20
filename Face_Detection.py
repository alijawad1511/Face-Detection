import cv2


trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)


while True:
    # To get Frames of Video
    successful_frame_read, frame = webcam.read()

    # Convert Frame to Gray Scale
    grayscalled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find Coordinates of Face
    face_coordinates = trained_face_data.detectMultiScale(grayscalled_img)

    # Draw Rectangles around Faces in Image
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Display Image
    cv2.imshow('Face Detection',frame)
    key = cv2.waitKey(1)

    # Press 'Q' to Exit
    if key==81 or key==113:
        break

# Release the Video Capture
webcam.release()
