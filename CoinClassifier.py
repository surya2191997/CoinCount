#Simple softmax classifier. Gray Image resized to 30 X 30, Flattened to a 900 dimensional vector.



import numpy as np
import os
import tensorflow as tf
import cv2

## TRAIN DATA ##
image_dim=900
numImages=7
numImagesTest=7
num_of_classes=4
i=0
train_set_img = np.zeros((numImages, image_dim), dtype=np.float64)
train_set_label = np.zeros((numImages,num_of_classes), dtype=np.float64)
test_set_img = np.zeros((numImagesTest, image_dim), dtype=np.float64)
test_set_label = np.zeros((numImagesTest,num_of_classes), dtype=np.float64)


for fname in os.listdir('/home/surya/Desktop/coin/train'):
    image=cv2.imread('/home/surya/Desktop/coin/train/'+fname,0)
    imgresize = cv2.resize(image, (30, 30))     
    data = np.array(imgresize)  
    imageflat=data.flatten()    
    train_set_img[i] = imageflat
        
	
    if fname[0] == '1': 
	   train_set_label[i][0]=1
    elif fname[0] == '2': 
	   train_set_label[i][1]=1
    elif fname[0] == '5': 
	   train_set_label[i][2]=1
    elif fname[0] == '0': 
	   train_set_label[i][3]=1
	
    i=i+1

i=0

## TEST DATA ##
for fname in os.listdir('/home/surya/Desktop/coin/train'):
    image=cv2.imread('/home/surya/Desktop/coin/train/'+fname,0)
    imgresize = cv2.resize(image, (30, 30))     
    data = np.array(imgresize)  
    imageflat=data.flatten()    
    test_set_img[i] = imageflat
    
    if fname[0] == '1': 
       test_set_label[i][0]=1
    elif fname[0] == '2': 
       test_set_label[i][1]=1
    elif fname[0] == '5': 
       test_set_label[i][2]=1
    elif fname[0] == '0': 
       test_set_label[i][3]=1
    
    i=i+1


		






## SOFTMAX CLASSIFIER ##

# Create the model
x = tf.placeholder(tf.float32, [None, 900])
W = tf.Variable(tf.zeros([900, 4]))
b = tf.Variable(tf.zeros([4]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 4])

 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
# Train Step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Training Epochs
for i in range(26):
    batch_xs = train_set_img
    batch_ys = train_set_label
    _, loss= sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys })
    print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss)) 
    

test_accuracy = sess.run(accuracy, feed_dict = {x: test_set_img, y_: test_set_label})
print("accuracy=", "{:.9f}".format(test_accuracy))




