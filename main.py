import cv2


img = cv2.imread('People/toni_stark__1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces.xml')
eye = cv2.CascadeClassifier('eye.xml')

results = faces.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
results_eye = eye.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    img_gray_face = gray[y:y+h, x:x+w]
    eyes = eye.detectMultiScale(img_gray_face, 1.1, 10)
    for (ex, ey, ew, eh) in results_eye:
        #cv2.circle(img, center=(ex, ey), radius=10, color=(255, 0, 0), thickness=2)
        cv2.rectangle(img, (ex, ey), (ex + ew,  ey + eh), (255, 0, 0), 2)

cv2.imshow('Result', img)
cv2.waitKey(0)
