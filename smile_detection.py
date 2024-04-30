import cv2

cascade_face = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('resources/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gray_roi = gray[y:y+h, x:x+w]
        smiles = cascade_smile.detectMultiScale(
            gray_roi,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.putText(img, 'SMILING', (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


    

