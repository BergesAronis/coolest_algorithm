import argparse
import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def predict(image):

	# Grayscale image, remove noise and construct threshold image
	shifted = cv2.pyrMeanShiftFiltering(image, 11, 70)
	# cv2.imshow("Input", image)

	gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	channels = cv2.mean(thresh)
	if channels[0] > 127:
		thresh = cv2.bitwise_not(thresh)
		gray = cv2.bitwise_not(gray)
	# cv2.imshow("Thresh", thresh)

	#Get euclidian distances and construct markers
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,
		labels=thresh)


	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)


	#Iterate through markers and find prominent contours
	count = [0,0,0]
	for label in np.unique(labels):
		if label == 0:
			continue


		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255


		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		c = max(cnts, key=cv2.contourArea)

		#Classify contours
		if cv2.contourArea(c) > 2:
			poly = cv2.approxPolyDP(c, 0.03*cv2.arcLength(c, True), True)
			cv2.drawContours(image, [poly], 0, (0, 255, 0), 2)
			x = poly.ravel()[0]
			y = poly.ravel()[1]

			if len(poly) == 3:
				cv2.putText(image, "Triangle", (x,y), font, 0.6, (0, 0 , 255), 2)
				count[2] += 1
			elif len(poly) == 4:
				cv2.putText(image, "Square", (x,y), font, 0.6, (0, 0 , 255), 2)
				count[0] += 1
			else:
				cv2.putText(image, "Circle", (x,y), font, 0.6, (0, 0 , 255), 2)
				count[1] += 1



	# cv2.imshow("Output", image)
	# cv2.waitKey(0)
	return count

def test(directory):
	it = iter(os.listdir(directory))

	total_pics = 0
	total_correct_pics = 0

	total_shapes = 0
	total_correct_shapes = 0


	for file in it:
		if file.endswith(".jpg"):
			image = cv2.imread(str(directory) + "/" + str(file))
			prediction = predict(image)

			label_file = str(directory) + "/" + str(next(it))
			f = open(label_file, "r")

			raw_labels = f.read()
			labels = [int(raw_labels[2]), int(raw_labels[7]), int(raw_labels[12])]
			total_pics += 1
			total_shapes += sum(labels)

			correct_prediction = [min(labels[0], prediction[0]),
								  min(labels[1], prediction[1]),
								  min(labels[2], prediction[2])]

			total_correct_shapes += sum(correct_prediction)


			if prediction == labels:
				total_correct_pics += 1


	return [round((total_correct_shapes/total_shapes)*100,2), round((total_correct_pics/total_pics)*100,2)]


if __name__ == "__main__":
	font = cv2.FONT_HERSHEY_COMPLEX
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=False, help="path to input image")
	ap.add_argument("-t", "--test", required=False, help="path to test folder")
	args = vars(ap.parse_args())
	if (args["image"]):
		image = cv2.imread(args["image"])
		print(predict(image))
	if (args["test"]):
		print(test(args["test"]))
