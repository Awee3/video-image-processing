import cv2
import numpy as np
import os
import time
import threading
from tkinter import Tk, Button

def start_saving_images():
    global saving_images, start_time, last_save_time
    saving_images = True
    start_time = time.time()
    last_save_time = 0 
    print("Penyimpanan gambar dimulai selama 20 detik...")

def run_opencv():
    global saving_images, start_time, last_save_time
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Tidak dapat mengakses kamera")
        return
    
    output_folder = "dataset"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_index = 0
    saving_images = False  # Flag untuk menentukan apakah gambar sedang disimpan
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([120 - 20, 50, 50])
        upper_blue = np.array([120 + 20, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        mask = cv2.erode(mask, kernel, iterations=5)
        mask = cv2.dilate(mask, kernel, iterations=5)

        mask_inv = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
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
                cv2.imshow("Warped", output_warped)
                if saving_images:
                    current_time = time.time()
                    if current_time - last_save_time >= 1: 
                        timestamp = int(current_time)
                        image_path = os.path.join(output_folder, f"card_{timestamp}.jpg")
                        cv2.imwrite(image_path, output_warped)
                        print(f"Gambar disimpan: {image_path}")
                        last_save_time = current_time
                
                    if current_time - start_time >= 25:
                      saving_images = False
                      print("Penyimpanan gambar selesai.")

        
        cv2.imshow("Label", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = Tk()
root.title("Penyimpanan Gambar Kartu Remi")

save_button = Button(root, text="Simpan Gambar", command=start_saving_images)
save_button.pack()

thread = threading.Thread(target=run_opencv, daemon=True)
thread.start()

root.mainloop()