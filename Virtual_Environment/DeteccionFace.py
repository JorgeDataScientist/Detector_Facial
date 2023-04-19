import cv2
import time

def draw_faces(faces, image, color):
    # Dibuja un cuadro alrededor de cada cara detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 8)

# Cargamos el archivo XML con los datos del modelo Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Iniciamos la cámara
cap = cv2.VideoCapture(0)

# Comprobamos que la cámara se ha iniciado correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    # En cada iteración del bucle, leemos un nuevo cuadro de la cámara:
    ret, frame = cap.read()
    #imagen en escala de los grises
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Verificamos si el cuadro se ha leído correctamente:
    if not ret:
        print("No se puede recibir el cuadro de la cámara. Saliendo ...")
        break    
    # Detectar las caras y ojos
    detected_faces = face_cascade.detectMultiScale(grayScaleImage, scaleFactor=1.3, minNeighbors=4)
    detected_eyes = eyes_cascade.detectMultiScale(grayScaleImage, scaleFactor=1.3, minNeighbors=4)
    draw_faces(detected_faces, frame, (0,255,255))
    draw_faces(detected_eyes, frame, (0,0,255))
    # Mostramos el cuadro en una ventana llamada "Visor de Camara":
    cv2.imshow('WebCam Face Detection ', frame)
    # Esperamos 1 milisegundo (aproximadamente) para que se presione una tecla:
    key = cv2.waitKey(1)
    if key == 27:  # 27 es el valor ASCII para "esc"
        # Cuando se presiona la tecla "esc", salimos del bucle:
        break
    # Agregamos un pequeño retraso de 0.1 segundos para permitir que la cámara capture un nuevo cuadro antes de continuar con la siguiente iteración del bucle:
    time.sleep(0.1)  # Pequeño retraso para permitir que la cámara capture un nuevo cuadro

# Liberamos los recursos y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()
