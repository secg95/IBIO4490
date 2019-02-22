
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import os



#Preparativos para hacer una imagen hibrida
DataFolder = Path('/imgs')
os.chdir(DataFolder)

#Secargan dos imagene hibridas un pez y una vión
img1 =  cv2.imread('fish.bmp')
img2 =  cv2.imread('plane.bmp')

#codigo de ajuste de tamaño(para el futuro implementar en función, buscar función)
x = int(np.mean([img1.shape[1],img2.shape[1]]))
y = int(np.mean([img1.shape[0],img2.shape[0]]))
img1 = cv2.resize(img1,(x,y))
img2 = cv2.resize(img2,(x,y))


#Parametros del filtro -gaussiano
opacador= 2
Sigma = 10
Kernel = 5


LowPassFilter = cv2.GaussianBlur(img2 ,(5,5), sigmaX=5,sigmaY=5)


#truco 1 reducir la luminosidad de la imagen 
LowPassFilterTemp =  cv2.GaussianBlur((img1/opacador).astype('uint8'),(5,5), sigmaX=5,sigmaY=5)
HighPassFilter = (img1/opacador).astype('uint8') - LowPassFilterTemp


#truco 2 cambiar negativos por 0
#LowPassFilterTemp  = cv2.GaussianBlur(img1,(5,5), sigmaX=5,sigmaY=5)
#HighPassFilter = img1  -  LowPassFilterTemp 
#negativos = img1  <  LowPassFilterTemp 
#HighPassFilter[negativos]=0

#escritura de la imagen Hibrida.
Tarea = LowPassFilter + HighPassFilter
cv2.imwrite('Hibrida.png', Tarea)
plt.imshow(Tarea)

#Piramides---------------------
#Piramide Gaussiana
def Gaussian_pyramid(img):
    ListaImagenes = [img]
    while True:
        if(img.shape[0] <= 4 or img.shape[1] <= 4):
            break
        lower_reso = cv2.pyrDown(img)
        ListaImagenes.append(lower_reso)
        img = lower_reso
    
    return ListaImagenes





#Piramide Lapalciana

def Laplace_pyramid(img):
    ListaImagenes = [img]  
    while True:
        if(img.shape[0] <= 4 or img.shape[1] <= 4):
            break
        def substract(imag1,imag2):
            x = min(imag1.shape[1],imag1.shape[1])
            y = min(imag1.shape[0],imag1.shape[0])
            imag1 = cv2.resize(imag1,(x,y))
            imag2 = cv2.resize(imag2,(x,y))
            Laplace = imag1 - imag2
            negativos = imag1 < imag2
            Laplace[negativos] = 0
            return Laplace   
        lower_reso = cv2.pyrDown(img)
        ListaImagenes.append(substract(img,cv2.pyrUp(lower_reso)))
        img = lower_reso
    return ListaImagenes

########Grafica de toda la piramide
    def get_one_image(images):
        img_list = images
        padding = 20
        max_width = []
        max_height = 0
        for img in img_list:
            max_width.append(img.shape[1])
            max_height += img.shape[0]
        w = np.max(max_width)
        h = max_height + padding

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((h, w, 3), dtype=np.uint8)

        current_y = 0  # keep track of where your current image was last placed in the y coordinate
        for image in img_list:
            # add an image to the final array and increment the y coordinate
            final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
            current_y += image.shape[0]
        #cv2.imwrite('out.png', final_image)
        plt.imshow(final_image)




###Imagen blend
        
#sol y luna
moon = cv2.imread('moon.jpg')
#se recorta la luna para un mejor efecto
moon = moon[45:moon.shape[0]-63, :]
sol = cv2.imread('sol.jpg')

#De nuevo resizeo, buscar función en el futuro pues esto se implementó muchas veces
x = int(np.mean([moon.shape[1],sol.shape[1]]))
y = int(np.mean([moon.shape[0],sol.shape[0]]))
sol = cv2.resize(sol,(x,y))
moon = cv2.resize(moon,(x,y))


#se decolora la mitad de cada imagen
moon[0:moon.shape[0],0:int(moon.shape[1]/2)] = 0
sol[0:sol.shape[0],int(sol.shape[1]/2)+1:sol.shape[1]]=0

#se suman las imagenes
Tarea2pubnto = sol + moon
plt.imshow(Tarea2pubnto)



#finalmente se calculan ambas piramides y se realiza el proceso para 
#obtener una imgen blended
gaus = Gaussian_pyramid(Tarea2pubnto)
lap = Laplace_pyramid(Tarea2pubnto)
#en que nivel de la pirmide empieza es un parametro a ajustar
upsampl = gaus[5]
#buckle que genera la imagen blended
for i in range(1,5):
   Laplace = lap[5-i]
   x = int(np.mean([upsampl.shape[1],Laplace.shape[1]]))
   y = int(np.mean([upsampl.shape[0],Laplace.shape[0]]))
   upsampl = cv2.resize(upsampl,(x,y))
   Laplace = cv2.resize(Laplace,(x,y))
   temp = upsampl+ Laplace
   upsampl = temp
 
plt.imshow(temp)
cv2.imwrite('blend.png', temp)
##########################################33