from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from scipy import misc
from scipy import ndimage
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import time
from layers.core import *
from layers.activationFunction import *

topengIMG = ""
valueConv1 = 12
valueConv2 = 12
valueLearningRate = 0.00001

class App(Tk):
    def __init__(self,*args,**kwargs):
       Tk.__init__(self,*args,**kwargs)
       self.notebook = ttk.Notebook()
       self.add_tab()
       self.notebook.grid(row=0)
  
    def add_tab(self):
        tab1 = Training(self.notebook)
        tab3 = Identifying(self.notebook)
        self.notebook.add(tab1,text="Training")
        self.notebook.add(tab3,text="Identifying")
  
  
class Training(Frame):
    def __init__(self,name,*args,**kwargs):
        Frame.__init__(self,*args,**kwargs)
        self.TextTraining = Text(self)
        BtnTraining = Button(self, text="TRAINING", pady=5, padx=10, command=self.training)
        BtnTraining.grid(row=4, column=1, pady=10, padx=10)

        self.inputConv1 = Entry(self, width = 20)
        Label(self, text="Kernel Conv1:").grid(row=0, column=0)
        self.inputConv1.grid(row=0, column=1, pady=10, padx=10)
        self.inputConv1.insert(END, 12)

        self.inputConv2 = Entry(self, width = 20)
        Label(self, text="Kernel Conv2:").grid(row=1, column=0)
        self.inputConv2.grid(row=1, column=1, pady=10, padx=10)
        self.inputConv2.insert(END, 24)

        self.inputLearningRate = Entry(self, width = 20)
        Label(self, text="Learning Rate:").grid(row=2, column=0)
        self.inputLearningRate.grid(row=2, column=1, pady=10, padx=10)
        self.inputLearningRate.insert(END, float(0.00001))

        self.testingResult = Entry(self, width = 20)
        self.testingResult.config(state='readonly')
        Label(self, text="Testing Result ( % ):").grid(row=3, column=0)
        self.testingResult.grid(row=3, column=1, pady=10, padx=10)

        self.TextTraining.config(padx=10, pady=10, wrap=WORD, height=30, width=100, font=('Verdana','8'), relief='solid')
        self.TextTraining.grid(row=5, column=2)
        self.progres = ttk.Progressbar(self,orient=HORIZONTAL,length=400,mode='determinate')
        self.progres.grid(row=4, column=2)
        self.name = name

    def training(self):
        self.progres.start()

        valueConv1 = int(self.inputConv1.get())
        print(valueConv1)

        valueConv2 = int(self.inputConv2.get())
        print(valueConv2)

        valueLearningRate = float(self.inputLearningRate.get())
        print(valueLearningRate)

        images = np.load('data/INDEXGAMBAR/OTOMATIS/TRAINING/training.npy')
        labels = np.load('data/INDEXGAMBAR/OTOMATIS/TRAINING/training_label.npy')
        labels = labels.astype(int)

        testing_images = np.load('data/INDEXGAMBAR/MANUAL/TESTING/testing.npy')
        testing_labels = np.load('data/INDEXGAMBAR/MANUAL/TESTING/testing_label.npy')
        testing_labels = testing_labels.astype(int)

        batch_size = 2
        error = 0
        last_error = 0
        tipe_operasi = 'training'
        batas_toleransi = 0.00001
        learning_rate = valueLearningRate

        convolutionalLayer1 = ConvolutionalLayer([batch_size, 96, 96, 3], valueConv1, 5, 1, tipe_operasi, 'convolutionalLayer1')
        relu1               = Relu(convolutionalLayer1.output_shape)
        maxPoolingLayer1    = MaxPoolingLayer(relu1.output_shape)
        convolutionalLayer2 = ConvolutionalLayer(maxPoolingLayer1.output_shape, valueConv2, 3, 1, tipe_operasi, 'convolutionalLayer2')
        relu2               = Relu(convolutionalLayer2.output_shape)
        maxPoolingLayer2    = MaxPoolingLayer(relu2.output_shape)
        fullyConnectedLayer = FullyConnectedLayer(maxPoolingLayer2.output_shape, 7, tipe_operasi, 'fullyConnectedLayer')
        sf                  = Softmax(fullyConnectedLayer.output_shape)

        if tipe_operasi == 'training':
            print("",valueConv1,"-",valueConv2,"-",learning_rate) 
            print(time.strftime("Start: "+"%Y-%m-%d %H:%M:%S", time.localtime()))

            for epoch in range(5000):
                
                batch_loss = 0
                batch_acc = 0
                val_acc = 0
                val_acc_ba =0
                val_loss = 0
                
                # train
                train_acc = 0
                train_loss = 0
                for i in range(images.shape[0] // batch_size):
                    img = images[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
                    label = labels[i * batch_size:(i + 1) * batch_size]
                    convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img))
                    maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
                    convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out))
                    maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
                    fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out)
                    batch_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))
                    train_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))
            
                    for j in range(batch_size):
                        if np.argmax(sf.softmax[j]) == label[j]:
                            batch_acc += 1
                            train_acc += 1
            
                    sf.gradient()
                    convolutionalLayer1.gradient(relu1.gradient(maxPoolingLayer1.gradient(
                        convolutionalLayer2.gradient(relu2.gradient(maxPoolingLayer2.gradient(
                            fullyConnectedLayer.gradient(sf.eta)))))))
            
                    if i % 1 == 0:
                        fullyConnectedLayer.backward(lr=learning_rate)
                        convolutionalLayer2.backward(lr=learning_rate)
                        convolutionalLayer1.backward(lr=learning_rate)
                        
                        batch_loss = 0
                        batch_acc = 0
            
                print(time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
                epoch + 1, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

                self.TextTraining.insert(INSERT, ""+time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
                epoch + 1, train_acc / float(images.shape[0]), train_loss / images.shape[0])+"\n")

                if(epoch != 0):
                    self.progres["value"] = (100 - ((train_loss / float(images.shape[0])) * 100))

                my_app.update()

                last_error = error
                error = train_loss / images.shape[0]

                if abs(last_error - error) < batas_toleransi and epoch != 0:
                    break

            convolutionalLayer1.simpan()
            convolutionalLayer2.simpan()
            fullyConnectedLayer.simpan()

        # testing
        epoch = 1
        val_loss = 0

        # Testing
        batch_size = 2
        val_acc = 0
        for i in range(testing_images.shape[0] // batch_size):
            img = testing_images[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
            label = testing_labels[i * batch_size:(i + 1) * batch_size]
            convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img))
            maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
            convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out))
            maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
            fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out)
            val_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))

            for j in range(batch_size):
                if np.argmax(sf.softmax[j]) == label[j]:
                    val_acc += 1

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f" % (
                    epoch, val_acc / float(testing_images.shape[0])))

        akurasiTesting = (val_acc / float(testing_images.shape[0]))

        self.testingResult.config(state='normal')
        self.testingResult.insert(END, akurasiTesting)
        self.testingResult.config(state='disabled')

        self.progres.stop()


class Identifying(Frame):
    def __init__(self,name,*args,**kwargs):
        Frame.__init__(self,*args,**kwargs)
        self.TextData = Text(self)
        self.LabelData = Label(self)
        self.BtnBrowse = Button(self, text="BROWSE MASK", pady=5, padx=10, command=self.fileDialog)
        self.BtnBrowse.grid(row=0, column=2, pady=10, padx=10)
        self.BtnIdentifying = Button(self, text="IDENTIFYING MASK", pady=5, padx=10, command=self.Identifying)
        self.BtnIdentifying.grid(row=0, column=3, pady=10, padx=10)

        self.LabelData.config(width=30, height=10, text="TOPENG", borderwidth=2, relief="groove")
        self.LabelData.grid(row=2, column=2, sticky=W, pady=10, padx=10)

        self.TextData.config(padx=10, pady=10, wrap=WORD, height=20, width=60, font='Verdana', relief='solid')
        self.TextData.grid(row=2, column=3)
        self.name = name

    def fileDialog(self):
        global topengIMG
        self.TextData.delete('1.0', END)
        topengIMG = filedialog.askopenfilename(initialdir = "/", title = "select a file", filetype = (("jpeg", "*.jpg"), ("All Files", "*.*")))
        # print(topengIMG)
        load = Image.open(topengIMG)
        if((load.size[0] > 5000) | (load.size[1] > 5000)):
            heightnya = int(load.size[0]/14)
            weightnya = int(load.size[1]/14)
        elif((load.size[0] > 4500) | (load.size[1] > 4500)):
            heightnya = int(load.size[0]/13)
            weightnya = int(load.size[1]/13)
        elif((load.size[0] > 4000) | (load.size[1] > 4000)):
            heightnya = int(load.size[0]/12)
            weightnya = int(load.size[1]/12)
        elif((load.size[0] > 3500) | (load.size[1] > 3500)):
            heightnya = int(load.size[0]/11)
            weightnya = int(load.size[1]/11)
        elif((load.size[0] > 3000) | (load.size[1] > 3000)):
            heightnya = int(load.size[0]/10)
            weightnya = int(load.size[1]/10)
        elif((load.size[0] > 2500) | (load.size[1] > 2500)):
            heightnya = int(load.size[0]/9)
            weightnya = int(load.size[1]/9)
        elif((load.size[0] > 2000) | (load.size[1] > 2000)):
            heightnya = int(load.size[0]/8)
            weightnya = int(load.size[1]/8)
        elif((load.size[0] > 1500) | (load.size[1] > 1500)):
            heightnya = int(load.size[0]/6)
            weightnya = int(load.size[1]/6)
        elif((load.size[0] > 1000) | (load.size[1] > 1000)):
            heightnya = int(load.size[0]/5)
            weightnya = int(load.size[1]/5)
        elif((load.size[0] > 500) | (load.size[1] > 500)):
            heightnya = int(load.size[0]/4)
            weightnya = int(load.size[1]/4)
        else:
            heightnya = load.size[0]
            weightnya = load.size[1]

        load = load.resize((heightnya,weightnya))
        photo = ImageTk.PhotoImage(load)
        self.LabelData.config(width=weightnya, height=heightnya, image=photo, borderwidth=2, relief="groove")
        self.LabelData.grid(row=2, column=2, sticky=W, pady=10, padx=10)
        self.LabelData.image = photo

    def Identifying(self):
        dataTopeng = []
        batch_size = 1
        val_acc = 0
        val_loss = 0
        tipe_operasi = 'testing'

        # ==================================  Cropping =========================================

        lionImage = misc.imread(topengIMG)
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
        lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
        lion_gray_blurred = ndimage.gaussian_filter(lion_gray, sigma=1.4)

        def SobelFilter(img, direction):
            if(direction == 'x'):
                Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
                Res = ndimage.convolve(img, Gx)
            if(direction == 'y'):
                Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
                Res = ndimage.convolve(img, Gy)
            return Res

        def Normalize(img):
            img = img/np.max(img)
            return img

        gx = SobelFilter(lion_gray_blurred, 'x')
        gx = Normalize(gx)
        gy = SobelFilter(lion_gray_blurred, 'y')
        gy = Normalize(gy)

        Mag = np.hypot(gx,gy)
        Mag = Normalize(Mag)

        Gradient = np.degrees(np.arctan2(gy,gx))

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


        NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
        NMS = Normalize(NMS)
        
        def DoThreshHyst(img):
            highThresholdRatio = 0.15  #0.09
            lowThresholdRatio = 0.15     #0.1
            GSup = np.copy(img)
            h = int(GSup.shape[0])
            w = int(GSup.shape[1])
            highThreshold = np.max(GSup) * highThresholdRatio
            lowThreshold = highThreshold * lowThresholdRatio    
            x = 0.1
            oldx=0
            
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
            
            GSup = (GSup == 1) * GSup 

            return GSup

        Final_Image = DoThreshHyst(NMS)

        a=Final_Image

        gambar = Image.open(topengIMG).convert('RGB')
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

        topeng = im.resize((96,96))

        dataTopeng.append(np.array(topeng))
        temp_dataTopeng = np.array(dataTopeng)

        kernelConv = "convolutionalLayer1"
        kernelConvData = 'temp/' + kernelConv +'_lr0,001.npy'
        kernelConv1Data = np.load(kernelConvData)
        kernelConv1 = int(kernelConv1Data.shape[3])

        kernelConv = "convolutionalLayer2"
        kernelConvData = 'temp/' + kernelConv +'_lr0,001.npy'
        kernelConv2Data = np.load(kernelConvData)
        kernelConv2 = int(kernelConv2Data.shape[3])

        convolutionalLayer1 = ConvolutionalLayer([batch_size, 96, 96, 3], kernelConv1, 5, 1, tipe_operasi, 'convolutionalLayer1')
        relu1               = Relu(convolutionalLayer1.output_shape)
        maxPoolingLayer1    = MaxPoolingLayer(relu1.output_shape)
        convolutionalLayer2 = ConvolutionalLayer(maxPoolingLayer1.output_shape, kernelConv2, 3, 1, tipe_operasi, 'convolutionalLayer2')
        relu2               = Relu(convolutionalLayer2.output_shape)
        maxPoolingLayer2    = MaxPoolingLayer(relu2.output_shape)
        fullyConnectedLayer = FullyConnectedLayer(maxPoolingLayer2.output_shape, 7, tipe_operasi, 'fullyConnectedLayer')
        sf                  = Softmax(fullyConnectedLayer.output_shape)

        for i in range(1):
            img = temp_dataTopeng[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
            convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img, 'testing'))
            maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
            convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out, 'testing'))
            maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
            fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out, 'testing')
            prediksi = sf.predict(fullyConnectedLayer_out)

            if np.argmax(sf.softmax[0]) == 0:
                result = 0
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Sidakarya\n\nTopeng Sidakarya merupakan tokoh Brahmana Keling yang merupakan sebutan seorang pendeta dari Jawa Timur, topeng Sidakarya berfungsi untuk supaya pekerjaan atau upacara berlangsung serta selesai dengan baik dan selamat. Anugerah kesempurnaan dan kemakmuran dapat disaksikan pada pertunjukan topeng Sidakarya yakni secara simbolis seperti menghamburkan uang kepeng dan beras kuning (sekarura).")
            elif np.argmax(sf.softmax[0]) == 1:
                result = 1
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Dalem\n\nTopeng Dalem Arsa Wijaya adalah tokoh raja (dalem) yang mengenakan topeng putih atau hijau muda, lambang tokoh halus. Ia menarikan tarian yang menampilkan kewibawaan, keagungan, dan keindahan.")
            elif np.argmax(sf.softmax[0]) == 2:
                result = 2
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Keras\n\nTopeng Keras merupakan tokoh perdana menteri atau patih, bersifat kuat dan kasar. Mukanya merah, menandai seorang pemberani. Geraknya lebar dan meruang, menunjukkan ketegasan dan kekuatan, tetapi sekaligus berwibawa dan terkendali.")
            elif np.argmax(sf.softmax[0]) == 3:
                result = 3
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Tua\n\nTopeng Tua adalah melambangkan tokoh raja yang sudah tua dan pensiun dari tahtanya, namun masing ingin mengenang masa mudanya, terlihat dari gerakannya yang lambat dan kadang-kadang cepat, dengan menggunakan perawakan yang tenang, berwibawa dan bijaksana dalam bersikap seperti halnya seorang yang sudah tua.")
            elif np.argmax(sf.softmax[0]) == 4:
                result = 4
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Penasar\n\nTopeng Penasar punta (Kelihan) Merupakan tokoh kelihan dengan sosok yang serius dan banyak memaparkan tentang upacara yang dilaksanakan.")
            elif np.argmax(sf.softmax[0]) == 5:
                result = 5
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Wijil\n\nTopeng Wijil (Cenikan) merupakan tokoh cenikan dengan sosok yang lebih tenang dan kadang membawakan sedikit lelucon dalam pementasan, yang juga banyak memaparkan tentang upacara yang dilaksanakan, dengan Bahasa sehari-hari.")
            elif np.argmax(sf.softmax[0]) == 6:
                result = 6
                self.TextData.delete('1.0', END)
                self.TextData.insert(INSERT, "Topeng Bujuh\n\nTopeng Penamprat Bracuk (Bujuh) adalah tokoh patih yang bermuka cokelat, mulut bujuh dan berkumis, dengan gerak yang lucu dan bersemangat.")
            else:
                self.TextData.delete('1.0', END)  
  
my_app = App()
my_app.title("Balinesse Mask Identifier")
# my_app.geometry('1366x768')
# my_app.state('zoomed')
my_app.mainloop()