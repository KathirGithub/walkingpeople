import cv2
faceClassifier=cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Create our body classifier


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    imagegrey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceClassifier.detectMultiScale(imagegrey,1.3,5)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
