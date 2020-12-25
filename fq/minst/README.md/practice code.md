## practice code

```python
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()(base)
```

## save code

```python
import tensorflow as tf
import os
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path="./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------------load the model-------------')
    model.load_weights(checkpoint_save_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True)
history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1,
callbacks=[cp_callback])
model.summary()
```

## storage  parameters code

```python
import tensorflow as tf
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path="./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------------load the model-------------')
    model.load_weights(checkpoint_save_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True)
history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1,
callbacks=[cp_callback])
model.summary()
print(model.trainable_variables)
file=open('./weight.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.nnumpy())+'\n')
file.close()
```

## loss&acc code



```python
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path="./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------------load the model-------------')
    model.load_weights(checkpoint_save_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True)
history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1,
callbacks=[cp_callback])
model.summary()
print(model.trainable_variables)
file=open('./weight.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()
##################   show   #################
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

![](.\loss and acc.png)

## predict code

```python
from PIL import Image
import numpy as np
import tensorflow as tf
model_save_path='./checkpoint/mnist.ckpt'
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')])
model.load_weights(model_save_path)
preNum=int(input("input the number of test pictures:"))
for i in range(preNum):
    image_path=input("the path of test picture:")
    img=Image.open(image_path)
    img=img.resize((28,28),Image.ANTIALIAS)
    img_arr=np.array(img.convert('L'))
    for i in range(28):
        for j in range(28):
            if img_arr[i][j]<200:
                img_arr[i][j]=255
            else:
                img_arr[i][j]=0
    img_arr=img_arr/255.0
    x_predict=img_arr[tf.newaxis,...]
    result=model.predict(x_predict)
    pred=tf.argmax(result,axis=1)
    print('\n')
    tf.print(pred)
```

