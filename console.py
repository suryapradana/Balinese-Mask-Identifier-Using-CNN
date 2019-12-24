import numpy as np
import time
from layers.core import *
from layers.activationFunction import *
import matplotlib.pyplot as plt

images = np.load('data/INDEXGAMBAR/MANUAL/TRAINING/training.npy')
labels = np.load('data/INDEXGAMBAR/MANUAL/TRAINING/training_label.npy')
labels = labels.astype(int)

testing_images = np.load('data/INDEXGAMBAR/MANUAL/TESTING/testing.npy')
testing_images_otomatis = np.load('data/INDEXGAMBAR/OTOMATIS/TESTING/testing.npy')
testing_labels = np.load('data/INDEXGAMBAR/MANUAL/TESTING/testing_label.npy')
testing_labels = testing_labels.astype(int)

index_images = np.load('data/INDEX/TESTING/testing.npy')
index_images = index_images.astype(int)

batch_size = 2
error = 0
last_error = 0
tipe_operasi = 'training'
batas_toleransi = 0.00001

filterconv1 = 12
filterconv2 = 24

convolutionalLayer1 = ConvolutionalLayer([batch_size, 96, 96, 3], 12, 5, 1, tipe_operasi, 'convolutionalLayer1')
relu1               = Relu(convolutionalLayer1.output_shape)
maxPoolingLayer1    = MaxPoolingLayer(relu1.output_shape)
convolutionalLayer2 = ConvolutionalLayer(maxPoolingLayer1.output_shape, 24, 3, 1, tipe_operasi, 'convolutionalLayer2')
relu2               = Relu(convolutionalLayer2.output_shape)
maxPoolingLayer2    = MaxPoolingLayer(relu2.output_shape)
fullyConnectedLayer = FullyConnectedLayer(maxPoolingLayer2.output_shape, 7, tipe_operasi, 'fullyConnectedLayer')
sf                  = Softmax(fullyConnectedLayer.output_shape)

# print("",filterconv1,"-",filterconv2,"- otomatis - 0.00002") 

if tipe_operasi == 'training':   
    print("",filterconv1,"-",filterconv2,"- manual - 0.000001") 
    print(time.strftime("Start: "+"%Y-%m-%d %H:%M:%S", time.localtime()))

    for epoch in range(1000):
        learning_rate = 0.000001
    
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
                                time.localtime()) + "  epoch: %5d , train_acc: %.5f  avg_train_loss: %.5f" % (
                epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))
        temting = train_loss / images.shape[0]
        
        last_error = error
        error = train_loss / images.shape[0]

        if abs(last_error - error) < batas_toleransi and epoch != 0:
            break

    convolutionalLayer1.simpan()
    convolutionalLayer2.simpan()
    fullyConnectedLayer.simpan()



# else:

#     epoch = 1
#     val_loss = 0

#     # Testing
#     batch_size = 2
#     val_acc = 0
#     for i in range(testing_images.shape[0] // batch_size):
#         img = testing_images[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
#         label = testing_labels[i * batch_size:(i + 1) * batch_size]
#         convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img))
#         maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
#         convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out))
#         maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
#         fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out)
#         val_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))

#         for j in range(batch_size):
#             if np.argmax(sf.softmax[j]) == label[j]:
#                 val_acc += 1

#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f" % (
#                 epoch, val_acc / float(testing_images.shape[0])))




# Testing
print("manual\n")
epoch = 1
val_loss = 0
loss_sidakarya = 0
loss_dalem = 0
loss_keras = 0
loss_tua = 0
loss_penasar = 0
loss_wijil = 0
loss_bujuh = 0

image_loss_sidakarya = []
image_loss_dalem = []
image_loss_keras = []
image_loss_tua = []
image_loss_penasar = []
image_loss_wijil = []
image_loss_bujuh = []

batch_size = 2
val_acc = 0
for i in range(1):
    img = testing_images[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
    label = testing_labels[i * batch_size:(i + 1) * batch_size]
    index_image = index_images[i * batch_size:(i + 1) * batch_size]
    convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img))
    # for i in range(12):
    #     plt.imshow(convolutionalLayer1_out[0,:,:,i], cmap='gray')
    #     plt.show()
    maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
    convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out))
    # for i in range(24):
    #     plt.imshow(convolutionalLayer2_out[0,:,:,i], cmap='gray')
    #     plt.show()
    maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
    fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out)
    val_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))

    for j in range(batch_size):
        if np.argmax(sf.softmax[j]) == label[j]:
            val_acc += 1
        else:
            if(label[j] == 0):
                loss_sidakarya += 1
                image_loss_sidakarya.append(index_image[j])
            elif(label[j] == 1):
                loss_dalem += 1
                image_loss_dalem.append(index_image[j])
            elif(label[j] == 2):
                loss_keras += 1
                image_loss_keras.append(index_image[j])
            elif(label[j] == 3):
                loss_tua += 1
                image_loss_tua.append(index_image[j])
            elif(label[j] == 4):
                loss_penasar += 1
                image_loss_penasar.append(index_image[j])
            elif(label[j] == 5):
                loss_wijil += 1
                image_loss_wijil.append(index_image[j])
            else:
                loss_bujuh += 1
                image_loss_bujuh.append(index_image[j])

print("==================================================================")
print("loss sidakarya : ", loss_sidakarya)
print(image_loss_sidakarya)
print("\n")
print("loss dalem : ", loss_dalem)
print(image_loss_dalem)
print("\n")
print("loss keras : ", loss_keras)
print(image_loss_keras)
print("\n")
print("loss tua : ", loss_tua)
print(image_loss_tua)
print("\n")
print("loss penasar : ", loss_penasar)
print(image_loss_penasar)
print("\n")
print("loss wijil : ", loss_wijil)
print(image_loss_wijil)
print("\n")
print("loss bujuh : ", loss_bujuh)
print(image_loss_bujuh)
print("\n")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f" % (
            epoch, val_acc / float(testing_images.shape[0])))






#testing otomatis
# Testing
print("otomatis\n")
epoch = 1
val_loss = 0
loss_sidakarya = 0
loss_dalem = 0
loss_keras = 0
loss_tua = 0
loss_penasar = 0
loss_wijil = 0
loss_bujuh = 0

image_loss_sidakarya = []
image_loss_dalem = []
image_loss_keras = []
image_loss_tua = []
image_loss_penasar = []
image_loss_wijil = []
image_loss_bujuh = []

batch_size = 2
val_acc = 0
for i in range(testing_images_otomatis.shape[0] // batch_size):
    img = testing_images_otomatis[i * batch_size:(i+1) * batch_size, 0:96, 0:96, 0:3]
    label = testing_labels[i * batch_size:(i + 1) * batch_size]
    index_image = index_images[i * batch_size:(i + 1) * batch_size]
    convolutionalLayer1_out = relu1.forward(convolutionalLayer1.forward(img))
    maxPoolingLayer1_out = maxPoolingLayer1.forward(convolutionalLayer1_out)
    convolutionalLayer2_out = relu2.forward(convolutionalLayer2.forward(maxPoolingLayer1_out))
    maxPoolingLayer2_out = maxPoolingLayer2.forward(convolutionalLayer2_out)
    fullyConnectedLayer_out = fullyConnectedLayer.forward(maxPoolingLayer2_out)
    val_loss += sf.calc_loss(fullyConnectedLayer_out, np.array(label))

    for j in range(batch_size):
        if np.argmax(sf.softmax[j]) == label[j]:
            val_acc += 1
        else:
            if(label[j] == 0):
                loss_sidakarya += 1
                image_loss_sidakarya.append(index_image[j])
            elif(label[j] == 1):
                loss_dalem += 1
                image_loss_dalem.append(index_image[j])
            elif(label[j] == 2):
                loss_keras += 1
                image_loss_keras.append(index_image[j])
            elif(label[j] == 3):
                loss_tua += 1
                image_loss_tua.append(index_image[j])
            elif(label[j] == 4):
                loss_penasar += 1
                image_loss_penasar.append(index_image[j])
            elif(label[j] == 5):
                loss_wijil += 1
                image_loss_wijil.append(index_image[j])
            else:
                loss_bujuh += 1
                image_loss_bujuh.append(index_image[j])

print("==================================================================")
print("loss sidakarya : ", loss_sidakarya)
print(image_loss_sidakarya)
print("\n")
print("loss dalem : ", loss_dalem)
print(image_loss_dalem)
print("\n")
print("loss keras : ", loss_keras)
print(image_loss_keras)
print("\n")
print("loss tua : ", loss_tua)
print(image_loss_tua)
print("\n")
print("loss penasar : ", loss_penasar)
print(image_loss_penasar)
print("\n")
print("loss wijil : ", loss_wijil)
print(image_loss_wijil)
print("\n")
print("loss bujuh : ", loss_bujuh)
print(image_loss_bujuh)
print("\n")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f" % (
            epoch, val_acc / float(testing_images_otomatis.shape[0])))

