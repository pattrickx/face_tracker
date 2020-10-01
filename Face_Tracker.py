import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist

from io import BytesIO
from IPython.display import  clear_output, Image, display
from PIL import Image as Img
from pynput.mouse import Controller, Button


x_n=[]
y_n=[]
ar_boca_n =[]
# FACE = list(range(17, 68))
# FACE_COMPLETA = list(range(0, 68))
LABIO = list(range(48, 61))
# SOMBRANCELHA_DIRETA = list(range(17, 22))
# SOMBRANCELHA_ESQUERDA = list(range(22, 27))
OLHO_DIREITO = list(range(36,42))
OLHO_ESQUERDO = list(range(42,48))
# NARIZ = list(range(27,35))
# MANDIBULA = list(range(0,17))

classificador_dlib_68 = 'shape_predictor_68_face_landmarks.dat'
classificador_dlib = dlib.shape_predictor(classificador_dlib_68)
detector_face = dlib.get_frontal_face_detector()

def pontos_marcos_faciais(img):
    retangulos = detector_face(img,1)
    if len(retangulos) == 0:
        return None
    marcos = []
    for ret in retangulos:
        marcos.append(np.matrix([[p.x,p.y] for p in classificador_dlib(img,ret).parts()]))
    return marcos

def aspecto_razao_boca(pontos_boca):
    
    a = dist.euclidean(pontos_boca[3], pontos_boca[9])
    b = dist.euclidean(pontos_boca[2], pontos_boca[10])
    c = dist.euclidean(pontos_boca[4], pontos_boca[8])
    d = dist.euclidean(pontos_boca[0], pontos_boca[6])
    aspecto_razao = (a + b + c)/(3.0 * d)
    
    return aspecto_razao
def anotar_marcos_casca_convexa(imagem, marcos):
    retangulos = detector_face(imagem, 1)
    global x_n,y_n,ar_boca_n

    if len(retangulos) == 0:
        return None

    p = marcos[0][0]
    p1 = marcos[0][27]
    p2 = marcos[0][16]
    p3 = marcos[0][8]
    p4 = marcos[0][30]

    x= np.linalg.norm(p2-p)
    x= np.linalg.norm(p1-p)/x
    xleft=0.3
    xhight=0.70

    y=np.linalg.norm(p3-p1)
    y=np.linalg.norm(p4-p1)/y
    ytop=0.19
    ybottom=0.35

    if x<xleft:
        x=xleft
    if x>xhight:
        x=xhight
    x= int((x-xleft)/(xhight-xleft)*1360)

     
    if y<ytop:
        y=ytop
    if y>ybottom:
        y=ybottom
    y= int((y-ytop)/(ybottom-ytop)*1024)
    mouse = Controller()
    # print(x)
    media_movel=20
    if(len(x_n)>media_movel):
        x_n.pop(0)
        x_n.append(x)
    else:
        x_n.append(x)

    if(len(y_n)>media_movel):
        y_n.pop(0)
        y_n.append(y)
    else:
        y_n.append(y)

    xs=0
    for i in x_n:
        xs+=i
    ys=0
    for i in y_n:
        ys+=i
    x= xs/len(x_n)
    y= ys/len(y_n)
    mouse.position=(x,y)
    y= p3[0,1]-p1[0,1]


    boca = round(aspecto_razao_boca(marcos[0][LABIO]),3)

    if (boca>0.6):
        mouse.click(Button.left)
    return imagem

def padronizar_imagem(frame):
    
    marcos_faciais = pontos_marcos_faciais(frame)
    if marcos_faciais is not None:
        frame= anotar_marcos_casca_convexa(frame,marcos_faciais)
    return frame

def exibir_video(frame):
    img = Img.fromarray(frame, "RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    display(Image(data=buffer.getvalue()))
    clear_output(wait=True)

captura = cv2.VideoCapture(0)

try:
    while(True):
        captura_ok, frame = captura.read()
        if  captura_ok:
            frame = frame[100:500, 100:500]
            frame=cv2.flip(frame,1)
            frame = cv2.resize(frame, (100, 100)) 
            frame = padronizar_imagem(frame)
            frame = cv2.resize(frame, (400, 400)) 
            cv2.imshow("Video", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
except KeyboardInterrupt:
    captura.release()
    cv2.destroyAllWindows()
    print("Interrompido")
