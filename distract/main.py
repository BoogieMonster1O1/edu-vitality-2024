import cv2
import numpy as np
import math
from gaze_tracking import GazeTracking

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://172.17.2.62:8001/video_feed")

gaze = GazeTracking()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

analysis = []

# saved facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# create an instance of the Facial landmark Detector with the model
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)


# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])


def pose_estimate(img,landmarks):
    # Camera internals
    ans=""
    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    shape = np.array(landmarks, dtype=np.float32).astype(np.uint)

    image_points = np.array([
        shape[0][0][30],  # Nose tip
        shape[0][0][8],  # Chin
        shape[0][0][36],  # Left eye left corner
        shape[0][0][45],  # Right eye right corne
        shape[0][0][48],  # Left Mouth corner
        shape[0][0][54]  # Right mouth corner
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    # print("rotation vector:",rotation_vector)


    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        ang = int(math.degrees(math.atan(m)))
    except:
        ang = 90

    if -45<ang  and ang<45:
        ans=True
    else:
        ans=False
    print("Angle:",ang)
    print("ans:",ans)
    cv2.line(img, p1, p2, (0, 255, 255), 2)
    # x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)
    #
    # cv2.line(img, p1, p2, (0, 255, 255), 2)
    # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

    return ans


def detect_faces(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=2,
            minSize=(20, 20)
        )

        return faces,gray

def calculate_attention_score(analysis):
    # Define weights for each attention level
    weights = {
        "Distracted": 0,
        "Somewhat Focussed": 0.5,
        "Definitely Focussed Listening": 1
    }
    
    # Count occurrences of each attention level
    distraction_count = analysis.count("Distracted")
    somewhat_count = analysis.count("Somewhat Focused")
    focused_count = analysis.count("Definitely Focused Listening")
    
    # Calculate total weighted attention
    total_weighted_attention = (weights["Distracted"] * distraction_count +
                                weights["Somewhat Focused"] * somewhat_count +
                                weights["Definitely Focused Listening"] * focused_count)
    
    # Calculate the total possible attention score
    total_possible_attention = distraction_count + somewhat_count + focused_count
    
    # Calculate and return the attention score in range 0 to 1
    if total_possible_attention == 0:
        return 0
    attention_score = total_weighted_attention / total_possible_attention
    return attention_score

def is_attentive(attention_score):
    if(attention_score > 0.7):
            print("user was attentive")
    elif(attention_score > 0.5):
        print("user was somewhat attentive")
    else:
        print("user was not attentive")

# cap = cv2.VideoCapture(0)

cap.set(3,1280) # set Width
cap.set(4,720) # set Height
while True:
    ret, img = cap.read()
    faces,gray=detect_faces(img)
    gaze.refresh(img)
    attention=False
    img=gaze.annotated_frame()

    if gaze.is_center():
        attention=True
    boolout=False
    try:
        _, landmarks = landmark_detector.fit(gray, faces)
        # print("landmarks:\n",landmarks)
        boolout=pose_estimate(img,landmarks) and attention

        # for landmark in landmarks:
        #     for x, y in landmark[0]:
        #         # display landmarks on "image_cropped"
        #         # with white colour in BGR and thickness 1
        #         # cv2.circle(img, (int(x), int(y)), 1, (0, 255,255), 1)

    except Exception as e:
        print(e)

    if boolout:
        text="Definitely Focused Listening"
        print(text)
        analysis.append(text)
    elif attention:
        text = "Somewhat Focused"
        print(text)
        analysis.append(text)
    else:
        text = "Distracted"
        print(text)
        analysis.append(text)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.circle(img,(x,y),min(x+w,y+h),(0,255,0),2)
        if boolout:
            cv2.putText(img,text,org = (x, y+h+12),fontFace = cv2.FONT_ITALIC,fontScale = 0.5,color = (255, 0, 255))
        else:
            cv2.putText(img, text, org=(x, y + h + 12), fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=(0, 0, 255))

        cv2.putText(img, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('Webcam Feed',img)
    k = cv2.waitKey(30) & 0xff  # press 'ESC' to quit
    if k == 27:
        distraction_count = analysis.count("Distracted")
        somewhat_count = analysis.count("Somewhat Focused")
        focused_count = analysis.count("Definitely Focused Listening")
        attention_score = calculate_attention_score(analysis)
        print("Distracted for : ",0.3*distraction_count,"secs")
        print("Somewhat focused for : ",0.3*somewhat_count,"secs")
        print("Fully focused for : ",0.3*focused_count,"secs")
        print("attention score : ",attention_score)
        is_attentive(attention_score)
        break
cap.release()
cv2.destroyAllWindows()
