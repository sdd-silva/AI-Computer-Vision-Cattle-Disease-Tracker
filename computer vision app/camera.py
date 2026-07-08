import cv2
import mediapipe as mp


mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        break


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = face_detector.process(rgb)


    if results.detections:

        for detection in results.detections:

            box = detection.location_data.relative_bounding_box

            h,w,_ = frame.shape

            x=int(box.xmin*w)
            y=int(box.ymin*h)

            width=int(box.width*w)
            height=int(box.height*h)


            cv2.rectangle(
                frame,
                (x,y),
                (x+width,y+height),
                (0,255,0),
                2
            )


            cv2.putText(
                frame,
                "Cow detected",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )


    cv2.imshow(
        "AI Face Detector",
        frame
    )


    if cv2.waitKey(1)==ord("q"):
        break


camera.release()
cv2.destroyAllWindows()


