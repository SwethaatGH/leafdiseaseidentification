import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
train_folder=r'/Users/swetha/Downloads/leafdataset/dataset/train'
test_folder=r'/Users/swetha/Downloads/leafdataset/dataset/test'
for x in [train_folder,test_folder]:
    filepaths=[]
    labels=[] 
    classlist=sorted(os.listdir(x))
    for disclass in classlist:
        if '__' in disclass:
            label=disclass.split('__')[1]
            classpath=os.path.join(x,disclass)
            flist=sorted(os.listdir(classpath))
            for f in flist:
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)            
                labels.append(label)
    fseries=pd.Series(filepaths,name='filepaths')
    lseries=pd.Series(labels,name='labels')        
    if x==train_folder:
        df=pd.concat([fseries,lseries],axis=1)
    else:
        test_df=pd.concat([fseries,lseries],axis=1)
train_df, test_df=train_test_split(df,train_size=.9,shuffle=True,random_state=123,stratify=df['labels'])
classes=sorted(list(train_df['labels'].unique()))
count=len(classes)
print('No. of classes in the dataset:',count)
groups=train_df.groupby('labels')
print('{0} {1}'.format('class','no. of images'))
countlist=[]
classlist=[]
for label in sorted(list(train_df['labels'].unique())):
    group=groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0} {1}'.format(label,str(len(group))))
train_gen=ImageDataGenerator(rescale=None,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_gen=ImageDataGenerator(rescale=None,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=train_gen.flow_from_directory(train_folder,target_size=(128,128),batch_size=32,class_mode='categorical')
test_set=test_gen.flow_from_directory(test_folder,target_size=(128,128),batch_size=32,class_mode='categorical')
def display_images(gen):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())    
    images,labels=next(gen) 
    plt.figure(figsize=(15,15))
    length=len(labels)
    if length<10:
        ran=length
    else:
        ran=10 #printing just 10 images from dataset for the idea of it
    for i in range(ran):        
        plt.subplot(5,5,i + 1)
        image=images[i]/255       
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name,color='black',fontsize=8)
        plt.axis('off')
    plt.show()
'''
print("training images")
display_images(training_set)
print("testing images")
display_images(test_set)
'''
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(38,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
labels1=(training_set.class_indices)
labels2=(test_set.class_indices)
#increase no of epochs and no of steps in each epoch to improve accuracy of model
fitted_model = model.fit(training_set,steps_per_epoch=10,epochs=1,validation_data = test_set,validation_steps = 125)
model.summary()
label1=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
       'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
       'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)',
       'Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
       'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
       'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
       'Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot',
       'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold',
       'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
       'Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
def testing(path):
    test_image=image.load_img(path,target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    fresult=np.max(result)
    label2=label1[result.argmax()]
    print(f"Disease identified is {label2}")
#Enter full path of image of leaf with disease to be identified
path='/Users/swetha/Downloads/leafdataset/images_for_test/PotatoEarlyBlight4.JPG'
testing(path)

