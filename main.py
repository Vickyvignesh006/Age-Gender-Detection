import cv2
import asyncio

# Function for detecting faces and drawing face boxes
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Change box color to red
    return frame, bboxs

async def process_frame(faceNet, ageNet, genderNet, frame):
    frame, bbox = await asyncio.get_event_loop().run_in_executor(None, faceBox, faceNet, frame)
    tasks = [process_face(frame, face, ageNet, genderNet) for face in bbox]
    await asyncio.gather(*tasks)
    return frame

async def process_face(frame, face, ageNet, genderNet):
    face_img = frame[max(0, face[1] - padding):min(face[3] + padding, frame.shape[0] - 1),
                    max(0, face[0] - padding):min(face[2] + padding, frame.shape[1] - 1)]
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]

    label = "{},{}".format(gender, age)
    cv2.rectangle(frame, (face[0], face[1] - 30), (face[2], face[1]), (0, 0, 255), -1)
    cv2.putText(frame, label, (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)

# Initialize variables
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-5)', '(6-10)', '(11-14)', '(15-18)', '(19-25)', '(26-32)', '(33-38)', '(39-45)', '(46-55)',
           '(56-70)', '(71-100)']
genderList = ['Male', 'Female']
padding = 20

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    task = asyncio.ensure_future(process_frame(faceNet, ageNet, genderNet, frame))
    loop = asyncio.get_event_loop()
    frame = loop.run_until_complete(task)

    cv2.imshow("Vicky & Kp Age-Gender Detector", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()