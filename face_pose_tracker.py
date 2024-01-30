import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import time

st.title('Face Pose Tracker App')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('FaceMesh Sidebar')

app_mode = st.sidebar.selectbox(
    'App Mode',
    ['About','Video']
)

@st.cache(suppress_st_warning=True)

def webcam_face_pose_video():
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)

    i = 0
    # max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
    kpil, kpil2 = st.columns(2)

    with kpil:
        st.markdown('**Frame Rate**')
        kpil_text = st.markdown('0')

    with kpil2:
        st.markdown('**Detected Faces**')
        kpil2_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as face_mesh:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence)
       
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
       
            cam = cv.VideoCapture(1)
            stframe = st.empty()
       
       
            if not cam.isOpened():
                st.error("Error: Could not open webcam.")
                return
            else:
                st.success("Webcam opened successfully.")
       
            video_running = False
           
            def toggle_video_state():
                global video_running
                video_running = not video_running
           
            if st.button("Stop Video"):
                toggle_video_state()  

            head_pose_info = {}      
           
            while cam.isOpened():
                ret, image = cam.read()
       
                if not ret:
                    st.warning("Could not read frame from webcam. Please check your webcam connection.")
                    break
       
                start_time = time.time()
                image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
       
                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []
                face_count = 0
       
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
       
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
       
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])
       
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)
       
                        focal_length = 1 * img_w
       
                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])
       
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        ret, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv.Rodrigues(rot_vec)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)
       
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360
       
                        if y < -10:
                            text = "Looking Left"
                        elif y > 10:
                            text = "Looking Right"
                        elif x < -10:
                            text = "Looking Down"
                        elif x > 10:
                            text = "Looking Up"
                        else:
                            text = "Forward"
       
                        nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
       
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                        head_pose_info[f'Person {face_count}'] = text

                        text_x = int(lm.x * img_w) - 160
                        text_y = int(lm.y * img_h) - 110
       
                        cv.line(image, p1, p2, (255, 0, 0), 3)
       
                        cv.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv.putText(image, "y: " + str(np.round(y, 2)), (500, 1000), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv.putText(image, f'Person {face_count} : {text}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

       
                        # Calculate elapsed time
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        if elapsed_time > 0:
                            fps = 1 / elapsed_time
                        else:
                            fps = float('inf')  

       
                        if fps == float('inf'):
                            fps_text = "Infinity"
                        else:
                            fps_text = str(int(fps))  # Convert FPS to string

                        cv.putText(image, f'FPS: {fps_text}', (20, 450), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
       
                kpil_text.write(f"<h1 style='text-align: center; color:red;'>{(fps_text)}</h1>", unsafe_allow_html=True)
                kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)

                frame = cv.resize(image,(0,0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame,channels='BGR', use_column_width=True)
       
                if cv.waitKey(1) & 0xFF == ord('q'):
                        toggle_video_state()
            cam.release()  
            cv.destroyAllWindows()

def image_resize(image, width=None, height=None):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)

    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=cv.INTER_AREA)

    return resized

# About Page

if app_mode == 'About':
    st.markdown('''
                 \n
                - FacePose Tracker is an application designed to analyze and track the head poses of individuals in real-time using a webcam. It accurately determines if a person is looking left, right, up, down, or straight ahead.  \n
                - Change the app mode from left side bar to video and click on **Use Webcam** to start the camera.  
               
                - [Github](https://github.com/WiTheR60334/Face-Pose-Tracker/) \n
    ''')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if app_mode == 'Video':
    use_webcam = st.sidebar.button('Use Webcam')
    if not use_webcam:
        st.markdown('''
                 \n
                - Click on the "Use Webcam" button to start the camera. \n
    ''')
    else:
        webcam_face_pose_video()