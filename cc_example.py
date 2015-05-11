from character_classifier import *
import cv2

# Load binary image with text
im = cv2.imread("data/iraq_classified.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Load general-purpose character classifier
clf = CharacterClassifier()
clf.load("data/gpcc.pkl")

# Predict characters
ic = clf.predict_image(im)

# Produce an image showing the predicted character labels
labeled_im = ic.labeledimage(fontscale=0.7, thickness=2)
cv2.imwrite("data/iraq_characters.png", labeled_im)