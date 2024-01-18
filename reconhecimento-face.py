import cv2 
import dlib
import matplotlib as plt
import seaborn


imagem = cv2.imread("imagens\px-people.jpg")

imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

classificador = cv2.CascadeClassifier("classificadores\haarcascade_frontalface_default.xml")

faces = classificador.detectMultiScale(imagem_gray, 1.3, 5)
print(len(faces))

imagem_anotada = imagem.copy()
face_imagem = 0


for (x, y, w, h ) in faces:
    cv2.rectangle(imagem_anotada, (x,y), (x+w, y+h), (255, 255, 0), 4)
    face_imagem+=1
    imagem_roi = imagem[y:y+h, x:x+w]
    cv2.imwrite(f'imagens-capturadas/face-capturada-{face_imagem}.png', imagem_roi)
    
    




cv2.imshow('imagem', imagem_anotada)



cv2.waitKey(0)



