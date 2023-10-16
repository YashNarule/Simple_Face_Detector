import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    "C:\\Users\\Lenovo\\Downloads\\haarcascade_frontalface_default.xml"
)

webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # Draw rectangles around the faces
    for x, y, w, h in face_coordinates:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (randrange(256), randrange(256), randrange(256)),
            2,
        )
    # Display the image with the faces spotted
    cv2.imshow("Face Detector", frame)
    # Wait here in the code and listen for a key press
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()
print("code completed")
