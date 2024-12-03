# import cv2
# from ultralytics import YOLO
# import os

# # Load your YOLO model
# model = YOLO("best.pt")  # Path to your trained YOLO model file

# # Input image path
# input_image_path = "1.png"  # Replace with your image path

# # Output folder for saving detected text regions
# output_folder = "detected_text_regions"
# os.makedirs(output_folder, exist_ok=True)

# # Load the input image
# image = cv2.imread(input_image_path)

# # Run the model on the input image
# results = model(image)

# # Extract detection results
# for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
#     x_min, y_min, x_max, y_max = map(int, box)  # Bounding box coordinates
#     cropped = image[y_min:y_max, x_min:x_max]  # Crop the detected region
#     output_path = os.path.join(output_folder, f"text_region_{i + 1}.png")
#     cv2.imwrite(output_path, cropped)  # Save the cropped region

#     print(f"Detected text region saved: {output_path}")

# # Display the original image with bounding boxes
# annotated_image = results[0].plot()  # Plot the image with annotations
# cv2.imshow("Detected Text", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# import cv2
# from ultralytics import YOLO
# import os
# import easyocr

# # Load YOLO model
# model = YOLO("best.pt")  # Replace with your YOLO model path

# # Input image path
# input_image_path = "2.jpg"  # Replace with your image path

# # Output folder for detected text regions
# output_folder = "detected_text_regions"
# os.makedirs(output_folder, exist_ok=True)

# # Load the input image
# image = cv2.imread(input_image_path)

# # Run YOLO model on the input image
# results = model(image)

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=True)  # Specify the language and enable GPU if available

# # Prepare for storing extracted text
# extracted_texts = []

# # Process each detected region
# for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
#     x_min, y_min, x_max, y_max = map(int, box)  # Get bounding box coordinates
#     cropped = image[y_min:y_max, x_min:x_max]  # Crop the detected region

#     # Save the cropped image
#     output_path = os.path.join(output_folder, f"text_region_{i + 1}.png")
#     cv2.imwrite(output_path, cropped)
#     print(f"Detected text region saved: {output_path}")

#     # Use EasyOCR to extract text from the cropped region
#     result = reader.readtext(cropped, detail=0)  # Get the text only (without box details)
#     extracted_text = " ".join(result).strip()  # Combine extracted text
#     extracted_texts.append(f"Region {i + 1}: {extracted_text}")

# # Print all extracted text
# print("\nExtracted Text from Detected Regions:")
# for text in extracted_texts:
#     print(text)

# # Display the original image with bounding boxes
# annotated_image = results[0].plot()  # Annotate the image with bounding boxes
# cv2.imshow("Detected Text with Bounding Boxes", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









import cv2
from ultralytics import YOLO
import os
import easyocr

# Load YOLO model
model = YOLO("best.pt")  # Replace with your YOLO model path

# Input image path
input_image_path = "sample/w.png"  # Replace with your image path

# Output folder for detected text regions
output_folder = "detected_text_regions"
os.makedirs(output_folder, exist_ok=True)

# File to save the extracted text
output_text_file = "extracted_text.txt"

# Load the input image
image = cv2.imread(input_image_path)

# Run YOLO model on the input image
results = model(image)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Specify the language and enable GPU if available

# Prepare for storing extracted text
extracted_texts = []

# Process each detected region
for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
    x_min, y_min, x_max, y_max = map(int, box)  # Get bounding box coordinates
    cropped = image[y_min:y_max, x_min:x_max]  # Crop the detected region

    # Save the cropped image
    output_path = os.path.join(output_folder, f"text_region_{i + 1}.png")
    cv2.imwrite(output_path, cropped)
    print(f"Detected text region saved: {output_path}")

    # Use EasyOCR to extract text from the cropped region
    result = reader.readtext(cropped, detail=0)  # Get the text only (without box details)
    extracted_text = " ".join(result).strip()  # Combine extracted text
    extracted_texts.append(f"{extracted_text}\n")

# Save all extracted text to the file
with open(output_text_file, "w", encoding="utf-8") as file:
    file.writelines(extracted_texts)

print(f"\nExtracted text saved to: {output_text_file}")

# Display the original image with bounding boxes
annotated_image = results[0].plot()  # Annotate the image with bounding boxes
cv2.imshow("Detected Text with Bounding Boxes", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
