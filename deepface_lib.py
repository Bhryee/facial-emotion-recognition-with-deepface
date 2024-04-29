import cv2 as cv
from deepface import DeepFace

# xml dosyasının içinde insan yüzünü algılamak için eğitilmiş makine öğrenimi modeli bulunmaktadır.
face_cascade = cv.CascadeClassifier('../haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Burada DeepFace.analyze metodu çağırılır.
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    
    # result'un türünü ve içeriğini kontrol etmek için hata ayıklama çıktıları eklenir.
    print(type(result))
    print(result)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # insan yüzünü tespit ettiğini göstermek için yuvarlak şekil çizer.
        """
        center = (x + w // 2, y + h // 2)
        radius = w // 2 if w < h else h // 2
        cv.circle(frame, center, radius, (255, 0, 0), 3)
        """
        # insan yüzünü tespit ettiğini göstermek için dikdörtgen şekil çizer.
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # Eğer result bir liste ise, her bir yüz için döngü
        if isinstance(result, list):
            for face_result in result:
                emotion = face_result['dominant_emotion']
                # frame (insan yüzü) adındaki görüntü üzerinde duygunun ne olduğunu yazar.
                cv.putText(frame, emotion, (x, y - 10), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        else:
            emotion = result['dominant_emotion']
            cv.putText(frame, emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



