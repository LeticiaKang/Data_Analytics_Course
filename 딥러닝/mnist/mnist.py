#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
print (tf.__version__)


# In[12]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[13]:


model = tf.keras.Sequential(layers=[
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
    
])
model.summary()


# In[14]:


model.compile (optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[16]:


history = model.fit(X_train, y_train, epochs=5)


# In[19]:


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print ("\n 테스트 정확도:", test_acc)


# In[20]:


import matplotlib.pyplot as plt
import numpy as np 
predictions = model.predict(X_test)
pred = np.argmax(predictions[0])

print("예측값:{}, 실제값: {}".format(pred, y_test[0]))

plt.imshow(X_test[0])
plt.show()


# In[21]:


history.history


# In[22]:


plt.plot(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.legend(["loss", "accuracy"])
plt.show()


# In[ ]:




