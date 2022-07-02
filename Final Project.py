#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras')


# In[2]:


get_ipython().system('pip install tensorflow')


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import tqdm
import random
from keras.preprocessing.image import load_img
warnings.filterwarnings('ignore')


# In[15]:


input_path = []
label = []



for path in os.listdir("C:/Users/Paridhi/Desktop/Topics in AI/AI Final Project/train/"):
    if "cat" in path:
        label.append(0)
    else:
        label.append(1)
    
    input_path.append(os.path.join("C:/Users/Paridhi/Desktop/Topics in AI/AI Final Project/train/", path))
    
print(input_path[0], label[0])


# In[16]:


df = pd.DataFrame()
df['images'] = input_path
df['label'] = label
df = df.sample(frac=1).reset_index(drop=True)
df.head()


# In[17]:


# to display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label']==1]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]



for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Dogs')
    plt.axis('off')


# In[18]:


# to display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label']==0]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]



for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Cats')
    plt.axis('off')


# In[19]:


import seaborn as sns
sns.countplot(df['label'])


# In[20]:


df['label'] = df['label'].astype('str')
df.head()


# In[21]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)


# In[22]:


from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
    rescale = 1./255, # normalization of images
    rotation_range = 40, # augmention of images to avoid overfitting
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)



val_generator = ImageDataGenerator(rescale = 1./255)



train_iterator = train_generator.flow_from_dataframe(
    train,
    x_col='images',
    y_col='label',
    target_size=(128,128),
    batch_size=512,
    class_mode='binary'
)



val_iterator = val_generator.flow_from_dataframe(
    test,
    x_col='images',
    y_col='label',
    target_size=(128,128),
    batch_size=512,
    class_mode='binary'
)


# In[23]:


from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense



model = Sequential([
                Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
                MaxPool2D((2,2)),
                Conv2D(32, (3,3), activation='relu'),
                MaxPool2D((2,2)),
                Conv2D(64, (3,3), activation='relu'),
                MaxPool2D((2,2)),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(1, activation='sigmoid')
])


# In[24]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[26]:


history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)


# In[27]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()



loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()


# In[ ]:




