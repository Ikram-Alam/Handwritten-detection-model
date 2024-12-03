import cv2
from ultralytics import YOLO
import os

# Load your YOLO model
model = YOLO("best.pt")  # Path to your trained YOLO model file

# Input image path
input_image_path = "1.png"  # Replace with your image path

# Output folder for saving detected text regions
output_folder = "detected_text_regions"
os.makedirs(output_folder, exist_ok=True)

# Load the input image
image = cv2.imread(input_image_path)

# Run the model on the input image
results = model(image)

# Extract detection results
for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
    x_min, y_min, x_max, y_max = map(int, box)  # Bounding box coordinates
    cropped = image[y_min:y_max, x_min:x_max]  # Crop the detected region
    output_path = os.path.join(output_folder, f"text_region_{i + 1}.png")
    cv2.imwrite(output_path, cropped)  # Save the cropped region

    print(f"Detected text region saved: {output_path}")

# Display the original image with bounding boxes
annotated_image = results[0].plot()  # Plot the image with annotations
cv2.imshow("Detected Text", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
