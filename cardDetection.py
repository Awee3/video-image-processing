import cv2
import numpy as np
import os
import tensorflow as tf

# Function to load class name mapping
def load_class_mapping(mapping_file):
    if os.path.exists(mapping_file):
        class_names = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                index, name = line.strip().split(': ')
                class_names[int(index)] = name
        return class_names
    else:
        print(f"Mapping file '{mapping_file}' not found.")
        return None

def detect_card():
    class_mapping_file = 'model/class_mapping.txt'
    class_names = load_class_mapping(class_mapping_file)

    model = tf.keras.models.load_model('model/cnn_model_v5.h5')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        # Konversi ke HSV dan buat mask untuk mendeteksi warna biru
        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([120 - 20, 50, 50])
        upper_blue = np.array([120 + 20, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_inv = cv2.bitwise_not(mask)
        mask = cv2.erode(mask, kernel, iterations=4)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(contour)
            if len(approx) == 4:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

                pts_original = np.float32([point[0] for point in approx])
                pts_original = sorted(pts_original, key=lambda x: (x[1], x[0]))
                pts_original = np.float32([pts_original[0], pts_original[1], pts_original[2], pts_original[3]])
                width, height = 200, 300  
                pts_transformed = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
                matrix = cv2.getPerspectiveTransform(pts_original, pts_transformed)
                output_warped = cv2.warpPerspective(frame, matrix, (width, height))

                resized_warped = cv2.resize(output_warped, (128, 128))  # Resize to (128, 128)

                # Normalize the image if necessary
                resized_warped = resized_warped.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]

                # Make prediction
                prediction = model.predict(resized_warped[np.newaxis, ...])  # Add batch dimension
                confidence = np.max(prediction)
                predicted_class = np.argmax(prediction)  # Get the class with the highest probability
                
                class_name = class_names.get(predicted_class, "Unknown") if class_names else "Unknown"
                confidence_percentage = confidence * 100
                text = f'{class_name} ({confidence_percentage:.2f}%)'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        cv2.imshow("Label", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_card()