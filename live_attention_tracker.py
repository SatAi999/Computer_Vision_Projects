import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine = more accurate eyes
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = 1280, 720  # virtual screen size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape
    gaze_x, gaze_y = None, None

    if results.multi_face_landmarks:
        mesh_points = np.array(
            [(int(p.x * w), int(p.y * h)) for p in results.multi_face_landmarks[0].landmark]
        )

        # Landmarks for left & right iris (468–473)
        left_iris = mesh_points[468:473]
        right_iris = mesh_points[473:478]

        # Iris centers (average of iris landmarks)
        left_center = left_iris.mean(axis=0).astype(int)
        right_center = right_iris.mean(axis=0).astype(int)

        # Approximate gaze point = midpoint between both irises
        gaze_x = int((left_center[0] + right_center[0]) / 2)
        gaze_y = int((left_center[1] + right_center[1]) / 2)

        # Draw eye centers
        cv2.circle(frame, tuple(left_center), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

        # Map gaze point to "virtual screen"
        norm_x = gaze_x / w
        norm_y = gaze_y / h
        screen_point = (int(norm_x * screen_w), int(norm_y * screen_h))

        # Overlay red dot on actual frame (scaled back)
        cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

        # Debug: Show projected gaze coordinates
        cv2.putText(frame, f"Gaze: {screen_point}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Live Attention Tracker (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
