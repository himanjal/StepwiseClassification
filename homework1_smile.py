import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors,  X, y):
	sumd = np.zeros(y.shape)
	#print predictors
	for location in predictors:
		#print location
		r1, c1, r2, c2 = location
		diff = X[:,r1,c1] - X[:,r2,c2]
		diff[diff > 0] = 1
		diff[diff < 0] = 0
		sumd += diff

	mean = np.divide(sumd,len(predictors))

	mean[mean > 0.5] = 1
	mean[mean <= 0.5] = 0

	return fPC(y, mean)
		



def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    

	predictors = []
	bestAccuracy = []
	for i in range(0,5):
		bestAccuracy = 0
		bestLocation = None
		for r1 in range(0,24):
			for c1  in range(0,24):
				for r2  in range(0,24):
					for c2  in range(0,24):
						if (r1,c1) == (r2,c2):
							continue
						if (r1,c1,r2,c2) in predictors:
							print "leaving" , (r1,c1,r2,c2)
							continue

						#predictors.append(((r1,c1), (r2,c2)))
						getAccuracy = measureAccuracyOfPredictors(predictors +  list(((r1,c1, r2,c2),)), trainingFaces,trainingLabels)

						if getAccuracy > bestAccuracy:
							
							bestAccuracy = getAccuracy
							bestLocation = (r1,c1, r2,c2)
							print "update", getAccuracy, bestLocation

		print bestLocation
		predictors.append(bestLocation)
	
	r1, c1, r2, c2 = bestLocation

	show = False
	if show:
		# Show an arbitrary test image in grayscale
		im = testingFaces[0,:,:]
		fig,ax = plt.subplots(1)
		ax.imshow(im, cmap='gray')
		# Show r1,c1
		rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# Show r2,c2
		rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
		ax.add_patch(rect)
		# Display the merged result
		plt.show()

	


	return predictors


def drawFeatures(predictors, testingFaces):
	im = testingFaces[0,:,:]
	fig,ax = plt.subplots(1)
	ax.imshow(im, cmap='gray')
	# Show r1,c1
	for location in predictors:
		r1, c1, r2, c2  = location
		rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# Show r2,c2
		rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
		ax.add_patch(rect)
	# Display the merged result
	plt.show()


    
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

def trainModel(trainingFaces,trainingLabels,testingFaces, testingLabels):
	samples = [400,800,1200,1600,2000]

	predictors = []
	for sampleNo in samples:
		log("Training Model with {} image samples.".format(sampleNo))
		predictors = stepwiseRegression(trainingFaces[:sampleNo],trainingLabels[:sampleNo],testingFaces,testingLabels)

		log("predictors selected{}".format(predictors))

		log("Training Accuracy of model on Training Data")

		accuracy = measureAccuracyOfPredictors(predictors,trainingFaces,trainingLabels)

		log("Training accuracy with {} samples = {}".format(sampleNo, accuracy))
		
		accuracy = measureAccuracyOfPredictors(predictors,testingFaces,testingLabels)

		log("Testing Accuracy with {} samples = {}".format(sampleNo, accuracy))

	log("Final Accuracy of model = {}".format(accuracy))

	log("Training with all the samples completed.")

	log("Final Predictors selected:\n{}".format(predictors))

	

	drawFeatures(predictors,testingFaces)



def log(text):
	global logString
	logString = logString + text + "\n"
	print text


logString = ""
logging = True
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    #predictors = stepwiseRegression(trainingFaces[:2000],trainingLabels[:2000],testingFaces,testingLabels)
    #predictors = [(20, 7, 17, 7), (13, 5, 11, 13), (20, 17, 16, 17), (12, 19, 12, 13), (10, 5, 14, 7), (13, 5, 15, 16), (22, 6, 16, 6), (12, 5, 11, 13), (18, 14, 14, 17), (3, 8, 2, 19)]
    #print predictors
    trainModel(trainingFaces,trainingLabels,testingFaces,testingLabels)
    
    if logging:
    	with open("LogFile.txt", 'w') as fh:
    		fh.write(logString)
    		fh.close()
    

