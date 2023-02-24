# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the pre-trained R-CNN model
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load the input image
image = cv2.imread('input_image.jpg')

# Set the confidence threshold
conf_threshold = 0.5

# Set the non-maximum suppression threshold
nms_threshold = 0.4

# Resize the input image
resized_image = cv2.resize(image, (300, 300))

# Create a blob from the resized image
blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Set the input blob for the model
model.setInput(blob)

# Run forward pass on the model to get the detections
start_time = time.time()
detections = model.forward()
end_time = time.time()

print("Time taken to detect objects: {:.5f} seconds".format(end_time - start_time))

# Loop over all the detections and draw bounding boxes around the detected objects
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        class_id = int(detections[0, 0, i, 1])

        # Get the coordinates of the bounding box
        x1 = int(detections[0, 0, i, 3] * image.shape[1])
        y1 = int(detections[0, 0, i, 4] * image.shape[0])
        x2 = int(detections[0, 0, i, 5] * image.shape[1])
        y2 = int(detections[0, 0, i, 6] * image.shape[0])

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes], confidences, conf_threshold, nms_threshold)

# Loop over the indices and draw the final bounding boxes
for i in indices:
    i = i[0]
    (x1, y1, x2, y2) = boxes[i]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the output image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()