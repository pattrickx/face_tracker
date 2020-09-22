import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist

from io import BytesIO
from IPython.display import  clear_output, Image, display
from PIL import Image as Img
from pynput.mouse import Controller, Button

min_olho_esquerdo,min_olho_direito,= (100,100)
x_n=[]
y_n=[]
ar_olho_esq_n =[]
ar_olho_dir_n =[]
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

def aspecto_razao_olhos(pontos_olhos):
    
    a = dist.euclidean(pontos_olhos[1], pontos_olhos[5])
    b = dist.euclidean(pontos_olhos[2], pontos_olhos[4])
    c = dist.euclidean(pontos_olhos[0], pontos_olhos[3])
    aspecto_razao = (a + b)/(2.0 * c)
    
    return aspecto_razao
def aspecto_razao_boca(pontos_boca):
    
    a = dist.euclidean(pontos_boca[3], pontos_boca[9])
    b = dist.euclidean(pontos_boca[2], pontos_boca[10])
    c = dist.euclidean(pontos_boca[4], pontos_boca[8])
    d = dist.euclidean(pontos_boca[0], pontos_boca[6])
    aspecto_razao = (a + b + c)/(3.0 * d)
    
    return aspecto_razao
def anotar_marcos_casca_convexa(imagem, marcos):
    retangulos = detector_face(imagem, 1)
    global min_olho_esquerdo,min_olho_direito,x_n,y_n,ar_olho_dir_n,ar_olho_esq_n

    if len(retangulos) == 0:
        return None
    # for k, d in enumerate(retangulos):
    #     # print(f'Identificado rosto: {k}')
    #     cv2.rectangle(imagem,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,0),2)
    p=[]
    p1=[]
    p2=[]
    p3=[]
    p4=[]

    for idx, ret in enumerate(retangulos):
        marco = marcos[idx]
        
        pontos = cv2.convexHull(marco[LABIO])
        cv2.drawContours(imagem, [pontos], 0, (0,255,0), 1)
    for marco in marcos:
        for idx,ponto in enumerate(marco):
            # print(ponto)
            centro =(ponto[0,0],ponto[0,1])
            if idx==0:
                p=ponto
                # cv2.circle(imagem,centro,3,(255,255,0),-1)
                # cv2.putText(imagem,str(idx),centro, cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255), 1)
            if idx==27:
                p1=ponto
                # cv2.circle(imagem,centro,3,(255,255,0),-1)
                # cv2.putText(imagem,str(idx),centro, cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255), 1)
            if idx==16:
                p2=ponto
                # cv2.circle(imagem,centro,3,(255,255,0),-1)
                # cv2.putText(imagem,str(idx),centro, cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255), 1)
            if idx==8:
                p3=ponto
                # cv2.circle(imagem,centro,3,(255,255,0),-1)
                # cv2.putText(imagem,str(idx),centro, cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255), 1)
            if idx==30:
                p4=ponto
                # cv2.circle(imagem,centro,3,(255,255,0),-1)
                # cv2.putText(imagem,str(idx),centro, cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255), 1)
   
    # cv2.line(imagem,(p[0,0],p[0,1]),(p1[0,0],p1[0,1]),(0,255,0), 1)
    # cv2.line(imagem,(p1[0,0],p1[0,1]),(p2[0,0],p2[0,1]),(0,255,0), 1)
    # cv2.line(imagem,(p1[0,0],p1[0,1]),(p4[0,0],p4[0,1]),(0,255,0), 1)
    # cv2.line(imagem,(p4[0,0],p4[0,1]),(p3[0,0],p3[0,1]),(0,255,0), 1)
    x= np.linalg.norm(p2-p)
    x= np.linalg.norm(p1-p)/x
    xleft=0.3
    xhight=0.70

    y=np.linalg.norm(p3-p1)
    y=np.linalg.norm(p4-p1)/y
    ytop=0.19
    ybottom=0.35
    # cv2.putText(imagem,f'x= {round(x,3)}',(10,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), 1)
    # cv2.putText(imagem,f'y= {round(y,3)}',(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), 1)
    
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
    # def test(pontos_olhos1,pontos_olhos2):
    #     a1 = dist.euclidean(pontos_olhos1[1], pontos_olhos1[5])
    #     b1 = dist.euclidean(pontos_olhos1[2], pontos_olhos1[4])
    #     c1 = dist.euclidean(pontos_olhos1[0], pontos_olhos1[3])
    #     a2 = dist.euclidean(pontos_olhos2[1], pontos_olhos2[5])
    #     b2 = dist.euclidean(pontos_olhos2[2], pontos_olhos2[4])
    #     c2 = dist.euclidean(pontos_olhos2[0], pontos_olhos2[3])
    #     print('a: ',a1,' | ',a2)
    #     print('b: ',b1,' | ',b2)
    #     print()
    # test(marcos[0][OLHO_DIREITO],marcos[0][OLHO_ESQUERDO])
    # ar_olho_dir = round(aspecto_razao_olhos(marcos[0][OLHO_ESQUERDO]),3)
    # ar_olho_esq = round(aspecto_razao_olhos(marcos[0][OLHO_DIREITO]),3)
    boca = round(aspecto_razao_boca(marcos[0][LABIO]),3)
    # if ar_olho_esq< min_olho_esquerdo:
    #     min_olho_esquerdo=ar_olho_esq
    # if ar_olho_dir< min_olho_direito:
    #     min_olho_direito=ar_olho_dir
    # info_oe =f'olho dir: {ar_olho_esq} minimo: {min_olho_esquerdo}'
    # info_od =f'olho esq: {ar_olho_dir} minimo: {min_olho_direito}'
    # print(info_oe)
    # print(info_od)
    # cv2.putText(imagem,info_oe,(5,50), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), 1)
    # cv2.putText(imagem,info_od,(10,80), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), 1)
#   
    
    # if(len(ar_olho_dir_n)>media_movel):
    #     ar_olho_dir_n.pop(0)
    #     ar_olho_dir_n.append(ar_olho_dir)
    # else:
    #     ar_olho_dir_n.append(ar_olho_dir)
    # ar_olho_dir_s=0

    # for i in ar_olho_dir_n:
    #     ar_olho_dir_s+=i

    # ar_olho_dir= ar_olho_dir_s/len(ar_olho_dir_n)
    
    # if(len(ar_olho_esq_n)>media_movel):
    #     ar_olho_esq_n.pop(0)
    #     ar_olho_esq_n.append(ar_olho_esq)
    # else:
    #     ar_olho_esq_n.append(ar_olho_esq)
    # ar_olho_esq_s=0

    # for i in ar_olho_esq_n:
    #     ar_olho_esq_s+=i

    # ar_olho_esq= ar_olho_esq_s/len(ar_olho_esq_n)

    # info_oe =f'olho dir: {ar_olho_esq} olho esq: {ar_olho_dir}'
    
    print(boca)
   
    # if (ar_olho_dir>0.19 and ar_olho_esq<0.15):
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
