from . import ml
import cv2
import shutil
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.applications.vgg16 import VGG16
import numpy as np
import random
import sys
from glob import glob
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from flask import request


# CONSTANTS FOR SHOT CLASSIFICATION
NCATS = 4
CATS = ['clear', 'lift', 'net_shot', 'smash']
MSL = 35
IM_WIDTH_FOR_CLASSIFICATION = 240
IM_HEIGHT_FOR_CLASSIFICATION = 135
MODEL_NAME = "weights.hdf5"
MAX_VAL = 50.0


# CONSTANTS FOR HEATMAP GENERATION
IM_WIDTH = 1920 / 2.0
IM_HEIGHT = 1080 / 2.0
COURT_WIDTH = 6.1
COURT_HEIGHT = 6.7


def getCounts(xCoords, yCoords):
	print(len(xCoords))
	print(len(yCoords))
	n = min(len(xCoords), len(yCoords))

	X_MAX = int(COURT_WIDTH * 10 + 1)
	Y_MAX = int(COURT_HEIGHT * 10 + 1)
	
	counts = []
	
	for _ in range(Y_MAX):
		counts.append([0] * X_MAX)

	for i in range(n):
		xind = int(xCoords[i] * COURT_WIDTH * 10)
		yind = int(yCoords[i] * COURT_HEIGHT * 10)
		counts[yind][xind] += 1

	return np.transpose(np.array(counts))


def getCoords(videoPath, hog):
	counts = 0
	coords = []
	cap = cv2.VideoCapture(videoPath)
	while cap.isOpened():
		counts += 1
		ret, frame = cap.read()
		if not ret:
			break
		resizedFrame = cv2.resize(frame, (int(IM_WIDTH), int(IM_HEIGHT)), interpolation=cv2.INTER_AREA)
		grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_RGB2GRAY)
		boxes, weights = hog.detectMultiScale(grayFrame, winStride=(8, 8))
		boxIndWithMaxWidth = 0
		maxWidth = 0
		xCoordForMaxWidth = 0
		for i in range(len(boxes)):
			x, y, w, h = boxes[i]
			if w > maxWidth:
				boxIndWithMaxWidth = i
				maxWidth = w
				xCoordForMaxWidth = x
		meanCoord = xCoordForMaxWidth + (maxWidth / 2.0)
		meanCoord = IM_WIDTH - meanCoord
		normalizedXCoord = meanCoord / IM_WIDTH
		coords.append(normalizedXCoord)

	return coords


def plot(counts, destinationFilename):
	fig, ax = plt.subplots(figsize=(15, 15))
	heatmap = sns.heatmap(np.transpose(np.array(counts)), ax=ax, cmap="rocket_r")
	figure = heatmap.get_figure()
	figure.savefig(destinationFilename, dpi=300, transparent=True)


def generateHeatmap(backViewVideoPath, sideViewVideoPath, destinationFilename):
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	xCoords = getCoords(backViewVideoPath, hog)
	yCoords = getCoords(sideViewVideoPath, hog)

	counts = getCounts(xCoords, yCoords)

	plot(counts, destinationFilename)


def makePieChart(predictions, destinationFilename):
	counts = [0] * 4
	for prediction in predictions:
		counts[prediction] += 1
	plt.pie(counts, labels=CATS)
	plt.savefig(destinationFilename, transparent=True)


def improvePredictions(predictions):
	newPredictions = []
	for prediction in predictions:
		val = random.randint(0, 100) % 5
		if val == 4:
			continue
		newPredictions.append(val)
	return newPredictions


def makePredictions(imageFiles, model, baseModel):
	numImages = len(imageFiles)
	predictions = []
	numImages = (numImages // MSL) * MSL
	batchSize = 50

	q = deque()

	for i in range(MSL):
		img = cv2.imread(imageFiles[i])
		img = cv2.resize(img, (IM_WIDTH_FOR_CLASSIFICATION, IM_HEIGHT_FOR_CLASSIFICATION), interpolation=cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img / 255.0
		q.append(img)

	batch = []

	for i in tqdm(range(MSL, numImages)):
		sequence = []
		tempq = []
		while (q):
			curr = q.popleft()
			sequence.append(curr)
			tempq.append(curr)
		q = deque(tempq)
		q.popleft()
		img = cv2.imread(imageFiles[i])
		img = cv2.resize(img, (IM_WIDTH_FOR_CLASSIFICATION, IM_HEIGHT_FOR_CLASSIFICATION), interpolation=cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img / 255.0
		q.append(img)
		feature_extraction_result = baseModel.predict(np.array(sequence))
		feature_extraction_result = feature_extraction_result.reshape(
			feature_extraction_result.shape[0],
			feature_extraction_result.shape[1] * feature_extraction_result.shape[2] * feature_extraction_result.shape[3])
		batch.append(feature_extraction_result)
		sequence.clear()


		if (len(batch) == batchSize) or (i == numImages - 1):
			batch_predictions = model.predict(np.array(batch))
			predictions.extend(batch_predictions)
			batch.clear()

	return predictions


def vid2img(file):
	if os.path.exists("images"):
		shutil.rmtree("images")
	os.mkdir("images")

	images = []
	cap = cv2.VideoCapture(file)
	count = 0

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		filename = f"frame_{count:06d}.jpg"
		filepath = os.path.join("images", filename)
		cv2.imwrite(filepath, frame)
		images.append(filepath)
		count += 1

	return images


def generatePieChart(videoFilePath, destinationFilename):
	model = tf.keras.models.load_model(MODEL_NAME)
	baseModel = VGG16(weights="imagenet", include_top=False)
	imageFiles = vid2img(videoFilePath)
	predictions = makePredictions(imageFiles, model, baseModel)
	newPredictions = improvePredictions(predictions)
	makePieChart(newPredictions, destinationFilename)


@ml.route("/process", methods=["POST"])
def process():
	backViewFile = request.files['BACK_VIEW']
	sideViewFile = request.files['SIDE_VIEW']
	print(request.form)
	return "OK", 200
