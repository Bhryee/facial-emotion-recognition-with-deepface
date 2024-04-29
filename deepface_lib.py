"""
import cv2 as cv
from deepface import DeepFace

face_cascade = cv.CascadeClassifier('/Users/bahriyeisgor/Desktop/Psikoloji Projesi/haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(img_path = frame, actions=['emotion'], enforce_detection = False)

    print(type(result))
    print(result)


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.1, 4)

    for(x,y,w,h) in face:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 3)

    emotion = result['dominant_emotion']
    txt = str(emotion)

    cv.putText(frame, txt(50, 50), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break

    


cap.release()
cv.destroyAllWindows()
"""


import cv2 as cv
from deepface import DeepFace

#insan yüzünü algılamak için eğitilmiş makine öğrenimi modelidir.
face_cascade = cv.CascadeClassifier('/Users/bahriyeisgor/Desktop/Psikoloji Projesi/haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Burada DeepFace.analyze metodunu çağırıyoruz
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    
    # result'un türünü ve içeriğini kontrol etmek için hata ayıklama çıktıları ekleyin
    print(type(result))
    print(result)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        """
        center = (x + w // 2, y + h // 2)
        radius = w // 2 if w < h else h // 2
        cv.circle(frame, center, radius, (255, 0, 0), 3)
        """
        #cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # Eğer result bir liste ise, her bir yüz için döngü
        if isinstance(result, list):
            for face_result in result:
                emotion = face_result['dominant_emotion']
                #Çıktının nasıl görüneceğini sağlayan komut
                #cv.putText(frame, emotion, (x, y - 10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        else:
            emotion = result['dominant_emotion']
            cv.putText(frame, emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



