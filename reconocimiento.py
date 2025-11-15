#importar librerias
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

#acceder a la carpeta
path = "dbPersonas"
images = []
clases = []
lista = os.listdir(path)
#print(lista)

#variables
comp1 = 100

#leer los rostros de bd
for lis in lista:
    #leer las imagenes de los rostros
    imgdb = cv2.imread(f'{path}/{lis}')
    #almacenar imagen 
    images.append(imgdb)
    #almacenar nombre
    clases.append(os.path.splitext(lis)[0])

print(clases)

#función para codificar los rostros
def codrostros(images):
    listacod = []

    #iteramos
    for img in images:
        #correcion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #codificar imagen
        cod = fr.face_encodings(img)[0]
        #almacenar
        listacod.append(cod)

    return listacod

#hora de ingreso
def horario(nombre):
    #abrir archivo en modo lectura y escritura
    with open('Information.csv', 'r+') as h:
        #leer la informacion
        data = h.readline()
        #crear lista de nombres
        listanombres = []

        #iterar cada linea del doc
        for line in data:
            #buscar la entrada y diferenciar con ,
            entrada = line.split(', ')
            #almacenar los nombres
            listanombres.append(entrada[0])
        
        #verificar si se almaceno el nombre
        if nombre not in listanombres:
            #extraer informacion actual
            info = datetime.now()
            #extraer fecha
            fecha = info.strftime('%Y:%m:%d')
            #extraer hora
            hora = info.strftime('%H:%M:%S')

            #guardar la informacion
            h.writelines(f'\n{nombre}, {fecha}, {hora}')
            print(info)

#llamar la funcion
rostroscod = codrostros(images)

#realizar videocaptura
cap = cv2.VideoCapture(0)   # ← usa 0, no 2

while True:
    #leer los fotogramas
    ret, frame = cap.read()

    #si falla la cámara, evita el crash
    if not ret or frame is None:
        print("Error: no se pudo leer un frame de la cámara")
        continue

    #reducir las imagenes para mejor procesamiento
    frame2 = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    #conversion de color
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    #buscar los rostros
    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)

    #iterar
    for facecod, faceloc in zip(facescod, faces):
        comparacion = fr.compare_faces(rostroscod, facecod)
        simi = fr.face_distance(rostroscod, facecod)
        min = np.argmin(simi)

        if comparacion[min]:
            nombre = clases[min].upper()
            print(nombre)

            yi, xf, yf, xi = faceloc
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4

            indice = comparacion.index(True)

            if comp1 != indice:
                r = random.randrange(0, 255, 50)
                g = random.randrange(0, 255, 50)
                b = random.randrange(0, 255, 50)
                comp1 = indice

            if comp1 == indice:
                cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                cv2.rectangle(frame, (xi, yf - 35), (xf, yf), (r, g, b), -1)
                cv2.putText(frame, nombre.upper(), (xi + 6, yf - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                horario(nombre)

    cv2.imshow("Reconocimiento Facial", frame)

    #leer el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
