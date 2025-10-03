import os
import cv2

def predict_defect(input_data, is_frame=False):
    """
    input_data: path to uploaded image OR a video frame
    is_frame: True if input_data is a frame from video
    """
    if is_frame:
        img = input_data.copy()
    else:
        img = cv2.imread(input_data)

    # Dummy prediction
    label = "Crack Detected"

    h, w, _ = img.shape
    start_point = (50, 50)
    end_point = (w-50, h-50)
    color = (0, 0, 255)
    thickness = 3
    cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.putText(img, label, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if is_frame:
        return label, img
    else:
        output_path = os.path.join("static/uploads", "output.jpg")
        cv2.imwrite(output_path, img)
        return label, output_path
