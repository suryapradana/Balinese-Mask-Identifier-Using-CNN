from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def unison_shuffled_copies_2(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

label_sidakarya = np.zeros(50)
label_dalem = np.zeros(50)
label_keras = np.zeros(50)
label_tua = np.zeros(50)
label_penasar = np.zeros(50)
label_wijil = np.zeros(50)
label_bujuh = np.zeros(50)

data_sidakarya = []
data_dalem = []
data_keras = []
data_tua = []
data_penasar = []
data_wijil = []
data_bujuh = []

index_sidakarya = 0
index_dalem = 0
index_keras = 0
index_tua = 0
index_penasar = 0
index_wijil = 0
index_bujuh = 0

for i in range(350):
    img_name = '.JPG'
    img_name = str(i) + img_name
    #topeng = Image.open('../data/topeng/'+img_name).convert('RGB')
    # topeng = cv2.imread('../data/topeng/'+img_name)

    # ==================================  Cropping =========================================

    imageUrl = "data/IMAGES/all/"+img_name

    # Load image into variable and display it
    lionImage = misc.imread(imageUrl) # Paste address of image
    if((lionImage.shape[0] > 5000) | (lionImage.shape[1] > 5000)):
        heightnya = int(lionImage.shape[0]/6)
        weightnya = int(lionImage.shape[1]/6)
    elif((lionImage.shape[0] > 4000) | (lionImage.shape[1] > 4000)):
        heightnya = int(lionImage.shape[0]/5)
        weightnya = int(lionImage.shape[1]/5)
    elif((lionImage.shape[0] > 3000) | (lionImage.shape[1] > 3000)):
        heightnya = int(lionImage.shape[0]/4)
        weightnya = int(lionImage.shape[1]/4)
    elif((lionImage.shape[0] > 2000) | (lionImage.shape[1] > 2000)):
        heightnya = int(lionImage.shape[0]/3)
        weightnya = int(lionImage.shape[1]/3)
    elif((lionImage.shape[0] > 1000) | (lionImage.shape[1] > 1000)):
        heightnya = int(lionImage.shape[0]/2)
        weightnya = int(lionImage.shape[1]/2)
    else:
        heightnya = lionImage.shape[0]
        weightnya = lionImage.shape[1]

    lion = misc.imresize(lionImage, (heightnya, weightnya))
    plt.imshow(lion, cmap = plt.get_cmap('gray'))
    plt.show()

    #print(lion.shape[1])

    # Convert color image to grayscale to help extraction of edges and plot it
    lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
    #lion_gray = lion_gray.astype('int32')
    # plt.imshow(lion_gray, cmap = plt.get_cmap('gray'))
    # plt.show()


    # Blur the grayscale image so that only important edges are extracted and the noisy ones ignored
    lion_gray_blurred = ndimage.gaussian_filter(lion_gray, sigma=1.4) # Note that the value of sigma is image specific so please tune it
    # plt.imshow(lion_gray_blurred, cmap = plt.get_cmap('gray'))
    # plt.show()


    # Apply Sobel Filter using the convolution operation
    # Note that in this case I have used the filter to have a maximum amgnitude of 2, but it can also be changed to other numbers for aggressive edge extraction
    # For eg [-1,0,1], [-5,0,5], [-1,0,1]
    def SobelFilter(img, direction):
        if(direction == 'x'):
            Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
            Res = ndimage.convolve(img, Gx)
            #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
        if(direction == 'y'):
            Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
            Res = ndimage.convolve(img, Gy)
            #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)
        
        return Res

    # Normalize the pixel array, so that values are <= 1
    def Normalize(img):
        #img = np.multiply(img, 255 / np.max(img))
        img = img/np.max(img)
        return img

    # Apply Sobel Filter in X direction
    gx = SobelFilter(lion_gray_blurred, 'x')
    gx = Normalize(gx)
    # plt.imshow(gx, cmap = plt.get_cmap('gray'))
    # plt.show()


    # Apply Sobel Filter in Y direction
    gy = SobelFilter(lion_gray_blurred, 'y')
    gy = Normalize(gy)
    # plt.imshow(gy, cmap = plt.get_cmap('gray'))
    # plt.show()

    # Apply the Sobel Filter using the inbuilt function of scipy, this was done to verify the values obtained from above
    # Also differnet modes can be tried out for example as given below:
    #dx = ndimage.sobel(lion_gray_blurred, axis=1, mode='constant', cval=0.0)  # horizontal derivative
    #dy = ndimage.sobel(lion_gray_blurred, axis=0, mode='constant', cval=0.0)  # vertical derivative

    dx = ndimage.sobel(lion_gray_blurred, axis=1) # horizontal derivative
    dy = ndimage.sobel(lion_gray_blurred, axis=0) # vertical derivative

    # Plot the derivative filter values obtained using the inbuilt function
    # plt.subplot(121)
    # plt.imshow(dx, cmap = plt.get_cmap('gray'))
    # plt.subplot(122)
    # plt.imshow(dy, cmap = plt.get_cmap('gray'))
    # plt.show()

    # Calculate the magnitude of the gradients obtained
    Mag = np.hypot(gx,gy)
    Mag = Normalize(Mag)
    # plt.imshow(Mag, cmap = plt.get_cmap('gray'))
    # plt.show()

    # Calculate the magnitude of the gradients obtained using the inbuilt function, again done to verify the correctness of the above value
    mag = np.hypot(dx,dy)
    mag = Normalize(mag)
    # plt.imshow(mag, cmap = plt.get_cmap('gray'))
    # plt.show()

    # Calculate direction of the gradients
    gradient = np.degrees(np.arctan2(gy,gx))

    # Calculate direction of the Gradients
    Gradient = np.degrees(np.arctan2(gy,gx))

    # Do Non Maximum Suppression with interpolation to get a better estimate of the magnitude values of the pixels in the gradient direction
    # This is done to get thin edges
    def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
        NMS = np.zeros(Gmag.shape)
        
        for i in range(1, int(Gmag.shape[0]) - 1):
            for j in range(1, int(Gmag.shape[1]) - 1):
                if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                    yBot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                    yBot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
        
        return NMS

    # This is also non-maxima suppression but without interpolation i.e. the pixel closest to the gradient direction is used as the estimate
    def NonMaxSupWithoutInterpol(Gmag, Grad):
        NMS = np.zeros(Gmag.shape)
        for i in range(1, int(Gmag.shape[0]) - 1):
            for j in range(1, int(Gmag.shape[1]) - 1):
                if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                    if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                    if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                    if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                    if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0

        return NMS


    # Get the Non-Max Suppressed output
    NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
    NMS = Normalize(NMS)
    # plt.imshow(NMS, cmap = plt.get_cmap('gray'))
    # plt.show()

    # Get the Non-max suppressed output on the same image but using the image using the inbuilt sobel operator
    nms = NonMaxSupWithInterpol(mag, gradient, dx, dy)
    nms = Normalize(nms)
    # plt.imshow(nms, cmap = plt.get_cmap('gray'))
    # plt.show()


    # Double threshold Hysterisis
    # Note that I have used a very slow iterative approach for ease of understanding, a faster implementation using recursion can be done instead
    # This recursive approach would recurse through every strong edge and find all connected weak edges
    def DoThreshHyst(img):
        highThresholdRatio = 0.09  
        lowThresholdRatio = 0.1 
        GSup = np.copy(img)
        h = int(GSup.shape[0])
        w = int(GSup.shape[1])
        highThreshold = np.max(GSup) * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio    
        x = 0.1
        oldx=0
        
        # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
        while(oldx != x):
            oldx = x
            for i in range(1,h-1):
                for j in range(1,w-1):
                    if(GSup[i,j] > highThreshold):
                        GSup[i,j] = 1
                    elif(GSup[i,j] < lowThreshold):
                        GSup[i,j] = 0
                    else:
                        if((GSup[i-1,j-1] > highThreshold) or 
                            (GSup[i-1,j] > highThreshold) or
                            (GSup[i-1,j+1] > highThreshold) or
                            (GSup[i,j-1] > highThreshold) or
                            (GSup[i,j+1] > highThreshold) or
                            (GSup[i+1,j-1] > highThreshold) or
                            (GSup[i+1,j] > highThreshold) or
                            (GSup[i+1,j+1] > highThreshold)):
                            GSup[i,j] = 1
            x = np.sum(GSup == 1)
        
        GSup = (GSup == 1) * GSup # This is done to remove/clean all the weak edges which are not connected to strong edges
        
        return GSup



    # The output of canny edge detection 
    Final_Image = DoThreshHyst(NMS)
    # plt.imshow(Final_Image, cmap = plt.get_cmap('gray'))
    # plt.show()


    # The output of canny edge detection using the inputs obtaind using the inbuilt sobel operator
    # Notice that the output here looks better than the one above, this might be because of the low magnitude of filter value used in our implementation of the Sobel Operator
    # Changing the filter to a higher value leads to more aggressive edge extraction and thus a better output.
    final_image = DoThreshHyst(nms)
    # plt.imshow(final_image, cmap = plt.get_cmap('gray'))
    # plt.show()

    a=Final_Image

    gambar = Image.open(imageUrl).convert('RGB')
    im = gambar.resize((weightnya, heightnya))

    for j in range(lion.shape[0]):
        for i in range(lion.shape[1]):
            if a[j][i] == 1.0:
                a[j][i] = 255

    k1 = 0
    for j in range(int(lion.shape[0]*2/3)):
        if np.mean(a[j]) < a.shape[1] / 300 or np.mean(a[j]) > len(a[j])*2/5:
            k1 += 1
        else:
            break

    k2 = 0
    for j in range(1, int(lion.shape[0]*2/3)):
        if np.mean(a[lion.shape[0]-j]) < a.shape[1] / 300 or np.mean(a[lion.shape[0]-j]) > len(a[lion.shape[0]-j])*2/5:
            k2 += 1
        else:
            break

    k3 = 0
    for j in range(int(lion.shape[1]*2/3)):
        if np.mean(a[:, j]) < a.shape[0] / 300 or np.mean(a[:, j]) > len(a[j]) * 2 / 5.5:
            k3 += 1
        else:
            break

    k4 = 0
    for j in range(1, int(lion.shape[1]*2/3)):
        if np.mean(a[:, lion.shape[1]-j]) < a.shape[0] / 300 or np.mean(a[:, lion.shape[1]-j]) > len(a[:, lion.shape[1]-j])*2/5:
            k4 += 1
        else:
            break

    im = im.crop((k3, k1, im.size[0]-k4, im.size[1]-k2))

    topeng = im.resize((86,86))

    plt.imshow(topeng)
    plt.show()


    # =================================== End Cropping ========================================

    
    if i < 50:
        label_sidakarya[index_sidakarya] = 1
        index_sidakarya += 1
        
        data_sidakarya.append(topeng)
    elif i < 100:
        label_dalem[index_dalem] = 2
        index_dalem += 1
        
        data_dalem.append(topeng)
    elif i < 150:
        label_keras[index_keras] = 3
        index_keras += 1
        
        data_keras.append(topeng)
    elif i < 200:
        label_tua[index_tua] = 4
        index_tua += 1
        
        data_tua.append(topeng)
    elif i < 250:
        label_penasar[index_penasar] = 5
        index_penasar += 1
        
        data_penasar.append(topeng)
    elif i < 300:
        label_wijil[index_wijil] = 6
        index_wijil += 1
        
        data_wijil.append(topeng)
    else:
        label_bujuh[index_bujuh] = 7
        index_bujuh += 1
        
        data_bujuh.append(topeng)
    
    
temp_sidakarya = np.array(data_sidakarya)
temp_dalem = np.array(data_dalem)
temp_keras = np.array(data_keras)
temp_tua = np.array(data_tua)
temp_penasar = np.array(data_penasar)
temp_wijil = np.array(data_wijil)
temp_bujuh = np.array(data_bujuh)

temp_sidakarya, label_sidakarya = unison_shuffled_copies_2(temp_sidakarya, label_sidakarya)
temp_dalem, label_dalem = unison_shuffled_copies_2(temp_dalem, label_dalem)
temp_keras, label_keras = unison_shuffled_copies_2(temp_keras, label_keras)
temp_tua, label_tua = unison_shuffled_copies_2(temp_tua, label_tua)
temp_penasar, label_penasar = unison_shuffled_copies_2(temp_penasar, label_penasar)
temp_wijil, label_wijil = unison_shuffled_copies_2(temp_wijil, label_wijil)
temp_bujuh, label_bujuh = unison_shuffled_copies_2(temp_bujuh, label_bujuh)

#splitting data training and testing
#====== testing
testing_sidakarya = temp_sidakarya[0:20,:,:,:]
testing_dalem = temp_dalem[0:20,:,:,:]
testing_keras = temp_keras[0:20,:,:,:]
testing_tua = temp_tua[0:20,:,:,:]
testing_penasar = temp_penasar[0:20,:,:,:]
testing_wijil = temp_wijil[0:20,:,:,:]
testing_bujuh = temp_bujuh[0:20,:,:,:]

testing_sidakarya_l = label_sidakarya[0:20]
testing_dalem_l = label_dalem[0:20]
testing_keras_l = label_keras[0:20]
testing_tua_l = label_tua[0:20]
testing_penasar_l = label_penasar[0:20]
testing_wijil_l = label_wijil[0:20]
testing_bujuh_l = label_bujuh[0:20]

#====== training
training = np.concatenate((temp_sidakarya[20:,:,:,:], temp_dalem[20:,:,:,:]), axis=0)
training = np.concatenate((training, temp_keras[20:,:,:,:]), axis=0)
training = np.concatenate((training, temp_tua[20:,:,:,:]), axis=0)
training = np.concatenate((training, temp_penasar[20:,:,:,:]), axis=0)
training = np.concatenate((training, temp_wijil[20:,:,:,:]), axis=0)
training = np.concatenate((training, temp_bujuh[20:,:,:,:]), axis=0)

training_l = np.concatenate((label_sidakarya[20:],label_dalem[20:]),axis=0)
training_l = np.concatenate((training_l, label_keras[20:]),axis=0)
training_l = np.concatenate((training_l, label_tua[20:]),axis=0)
training_l = np.concatenate((training_l, label_penasar[20:]),axis=0)
training_l = np.concatenate((training_l, label_wijil[20:]),axis=0)
training_l = np.concatenate((training_l, label_bujuh[20:]),axis=0)

#shuffled before training
training, training_l = unison_shuffled_copies_2(training, training_l)

#SAVE
#training
np.save('../data/TRAINING/training', training)
np.save('../data/TRAINING/training_label', training_l)

#testing
np.save('../data/TESTING/testing_sidakarya', testing_sidakarya)
np.save('../data/TESTING/testing_sidakarya_label', testing_sidakarya_l)
np.save('../data/TESTING/testing_dalem', testing_dalem)
np.save('../data/TESTING/testing_dalem_label', testing_dalem_l)
np.save('../data/TESTING/testing_keras', testing_keras)
np.save('../data/TESTING/testing_keras_label', testing_keras_l)
np.save('../data/TESTING/testing_tua', testing_tua)
np.save('../data/TESTING/testing_tua_label', testing_tua_l)
np.save('../data/TESTING/testing_penasar', testing_penasar)
np.save('../data/TESTING/testing_penasar_label', testing_penasar_l)
np.save('../data/TESTING/testing_wijil', testing_wijil)
np.save('../data/TESTING/testing_wijil_label', testing_wijil_l)
np.save('../data/TESTING/testing_bujuh', testing_bujuh)
np.save('../data/TESTING/testing_bujuh_label', testing_bujuh_l)
