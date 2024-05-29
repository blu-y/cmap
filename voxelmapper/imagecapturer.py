import cv2
import datetime

def capture_image():
    cap = cv2.VideoCapture(2)
    print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture image')
            break
        cv2.imshow('Camera', frame)
        # Check if the Enter key is pressed
        key = cv2.waitKey(1)
        if key == 13:
            current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")[:-4]
            file_name = f"voxelmapper/images/{current_time}.png"
            cv2.imwrite(file_name, frame)
            print(frame.shape, 'Image saved!', file_name)
        # Check if the Esc key is pressed
        elif key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
capture_image()