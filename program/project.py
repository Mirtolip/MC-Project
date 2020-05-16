## Mirtolip Saitov
## MC Project
## Face Detection 
## video

import cv2

#name of the video
filename = '../videoplayback.mp4'
# capture frames from a file 
cap = cv2.VideoCapture(filename) 
# Create the haar cascade classifier
faceCascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

fourcc   = cv2.VideoWriter_fourcc(*'XVID')
out      = cv2.VideoWriter('../oututs/output.avi', fourcc, 20.0, (700,400))

while(True):
	# Capture frame 
	ret, frame = cap.read()
	frame = cv2.resize(frame, (int(700), int(400)))  

	# Convert Colorful frame to gray scale frame 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (int(x), 255, int(y)), 2)

	out.write(frame)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
