# coding: utf-8


import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import random
import operator
import matplotlib.pyplot as plt
import sys
from fastdtw import fastdtw  #dynamic time warping

#load sequences
sequences = []
for bat_number in range(4):
    with open("Data_Sequences_fits/Sequences_"+str(bat_number*500)+"_"+str(bat_number*500+500)+".df", "rb") as fp:   # Unpickling sequences
        sequences += pickle.load(fp)
    
#####Next we define some useful functions######

#function to pass sequence length (without padding) to the RNN
def length(sequence_batch):
    used = tf.sign(tf.reduce_max(tf.abs(sequence_batch), 3))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    length = tf.squeeze(length)
    return length

def plot(trace, title):
    t_len = len(trace)
    plt.figure(figsize=(10,4))
    plt.plot(np.linspace(0, (t_len-1)*0.04, t_len), trace[:,0]) # plotting CPU0
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Temp [C]")
    plt.xlim(0, (t_len-1)*0.04)
    plt.tight_layout()
    #plt.savefig(title+'.png', format='png', dpi=300)
    plt.show() 
  
#Compresses the per-timestamp labels to a sequence of tasks (E.g. [4,3,1,2,3,2])  
def Compress_labels(labels, threshold): 
    start = 0
    end = 0 
    curr = labels[0]
    count = 1
    compressed = []
    app_flag = 0
    for i in range(1, len(labels)): #traverse the labels one by one
        
        if labels[i] == curr: #check if label is equal to the previous one
            end += 1
            count += 1
            if count >= threshold: #consider a task detected when we have at least >threshold consecutive timestamps with that label. e.g. 4444444444
                app_flag = 1 #flag to signal we have detected at least one task
                if i == len(labels)-1:
                    if compressed:
                        if compressed[-1][0] != curr:   
                            compressed.append([curr,start,end])
                        else: compressed[-1][2] = end       #if e.g. the last detected task was 4 and then after a noisy patch we detect again 4, we just update the end time.
                    else: compressed.append([curr,start,end])
                
        else:
            if app_flag == 1: 
                if compressed:
                    if compressed[-1][0] != curr:
                        compressed.append([curr,start,end])
                    else:
                        compressed[-1][2] = end
                else: compressed.append([curr,start,end])
            app_flag = 0
            start = i
            end = i
            curr = labels[i]
            count = 1
    
    return compressed #return a list with the compressed labels, the start and the end of each task

#accuracy over compressed labels using dtw
def TaskAccuracy(preds, batch_labels):
    task_accs = []
    for i in range(preds.shape[0]):
        pred_symlist = np.asarray(Compress_labels(np.asarray(preds[i]), 10))
        target_symlist = np.asarray(Compress_labels(np.asarray(batch_labels[i]), 10))
        pred_tasks = pred_symlist[:,0]
        target_tasks = target_symlist[:,0]

        dist = lambda x,y: np.sign(abs(x-y))
        distance,_ = fastdtw(target_tasks, pred_tasks, dist=dist)
        task_accs.append(distance)
    task_accs = np.asarray(task_accs)
    task_accuracy = np.mean(task_accs)
    return task_accuracy
	
#timing error of compressed labels using dtw
def TimingError(preds, batch_labels):
    errors = []
    for i in range(preds.shape[0]):
        pred_symlist = np.asarray(Compress_labels(np.asarray(preds[i]), 10))
        target_symlist = np.asarray(Compress_labels(np.asarray(batch_labels[i]), 10))
        pred_ends = pred_symlist[:,2]
        target_ends = target_symlist[:,2]

        dist = lambda x,y: np.abs(x-y)
        distance,_ = fastdtw(target_ends, pred_ends, dist=dist)
        errors.append(distance)
    errors = np.asarray(errors)
    timing_error = np.mean(errors)
    return timing_error #in timestamps,  Note: This error is accumulated over an entire sequence. Divide by number of tasks in a sequence to get timing shift per task.

#define convLSTM cells
def single_cell(hidden):
    conv_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1,
                                           input_shape=[1,11],
                                           output_channels=hidden,
                                           kernel_shape=[3])
    return tf.nn.rnn_cell.DropoutWrapper(conv_cell,output_keep_prob=0.8)
	
	
########## Main program #################################	
	
#zero pad to max length=1250
padded = np.zeros((len(sequences),1250,11))
for i in range(len(sequences)):
    p = sequences[i]["sample"]
    l = p.shape[0]
    p = np.pad(p, ((0,1250-l),(0,0)), 'constant', constant_values=(0,0))
    padded[i] = p

#Get ground truth labels for every timestamp 
labels = np.zeros((len(padded),1250))
for i in range(len(padded)):
    tr_lens = sequences[i]['trace_lengths']
    n = sequences[i]['num_traces']
    label_dict = sequences[i]['labels']
    labels[i, 0:tr_lens[0]] = label_dict[0][0]
    k = tr_lens[0]
    for j in range(1,n):
        labels[i, k:k+tr_lens[j]] = label_dict[j][0]
        k = k + tr_lens[j]
    labels[i, k:] = 5

labels = labels.astype(int)

#design encoder-decoder model with tensorflow
tf.reset_default_graph()
num_layers = 2
n_classes = 6
hidden = 128
max_length = 1250
learning_rate = 0.001
batch_labels = tf.placeholder(tf.int32, shape=[None,max_length])
input_ = tf.placeholder(tf.float32, shape=[None, max_length, 11]) #[batch_size, sequence_length, sensors]
p_input = tf.expand_dims(input_, 2) #[batch, len, 1, 11]



batch_len = tf.shape(input_)[0]



mult_enc_cell = tf.contrib.rnn.MultiRNNCell([single_cell(hidden) for _ in range(num_layers)])

mult_dec_cell = tf.contrib.rnn.MultiRNNCell([single_cell(hidden) for _ in range(num_layers)])

with tf.variable_scope('encoder'):
    (enc_output, enc_state) = tf.nn.dynamic_rnn(mult_enc_cell, p_input, dtype=tf.float32, sequence_length=length(p_input)) #add sequence length with length function



with tf.variable_scope('decoder'):
            
            dec_inputs = tf.zeros(dtype=tf.float32, shape=[batch_len, max_length, 1, 11])
            (dec_outputs, dec_state) = tf.nn.dynamic_rnn(mult_dec_cell, dec_inputs, initial_state=enc_state,
                    dtype=tf.float32)
            dec_output_ = tf.squeeze(dec_outputs)
        
            
with tf.variable_scope('softmax'): #output layer to produce the desired output of labels
    weight_ = tf.Variable(tf.truncated_normal([hidden, n_classes], dtype=tf.float32), name='softmax_weight')
    bias_ = tf.Variable(tf.constant(0.1, shape=[n_classes], dtype=tf.float32), name='softmax_bias')
    weight_ = tf.tile(tf.expand_dims(weight_, 0), [batch_len, 1, 1])
    logits = tf.matmul(dec_output_, weight_) + bias_     

#sequence loss function which internally uses softmax and cross-entropy loss 
loss = tf.contrib.seq2seq.sequence_loss(logits, batch_labels, 
                                        tf.ones([batch_len, max_length]), # sample_weights
                                        average_across_batch=True,
                                        average_across_timesteps=False) 
total_loss = tf.reduce_sum(loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate)


max_grad_norm = 5.0
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
gradients = zip(grads, tvars)
training_op = optimizer.apply_gradients(gradients)


#Accuracy per-timestamp
preds = tf.argmax(logits,2,output_type=tf.int32)
equality = tf.equal(preds, batch_labels)
accuracy = tf.reduce_mean(tf.cast(equality, "float"))


#tensorboard
summ_loss = tf.summary.scalar('train_loss', total_loss)
summ_loss_val = tf.summary.scalar('test_loss', total_loss)
summ_acc = tf.summary.scalar('train_acc', accuracy)
summ_acc_val = tf.summary.scalar('test_acc', accuracy)

task_accuracy = tf.placeholder(tf.float32, shape=[])
summ_task_acc = tf.summary.scalar('train_task_dtw', task_accuracy)
summ_task_acc_val = tf.summary.scalar('test_task_dtw', task_accuracy)

timing_error = tf.placeholder(tf.float32, shape=[])
summ_time_error_val = tf.summary.scalar('test_time_dtw', timing_error)

init = tf.global_variables_initializer()




#Perform the training
n_epochs = 1001
batch_size = 256
from sklearn.utils import shuffle
padded, labels = shuffle(padded, labels, random_state=0)
X_train = padded[:1600]
Y_train = labels[:1600]
X_test = padded[1600:2000]
Y_test = labels[1600:2000]

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("logs/ConvLSTMseq2seq_DTW", sess.graph) #for tensorboard
    print("Training starting")
    sys.stdout.flush()
    for epoch in range(n_epochs):
        X_train_, Y_train_ = shuffle(X_train, Y_train, random_state=epoch)
        batches = [ X_train_[k:k+batch_size] for k in range(0, len(X_train_), batch_size) ]
        targets =  [ Y_train_[k:k+batch_size] for k in range(0, len(Y_train_), batch_size) ]
        for i in range(len(batches)):
            _, train_loss,train_acc = sess.run([training_op,summ_loss,summ_acc], feed_dict={input_: batches[i], batch_labels: targets[i]})
            writer.add_summary(train_loss, epoch)
            writer.add_summary(train_acc, epoch)
        
        if epoch % 10 == 0: #test
            test_preds, test_loss, test_acc, summ_test_loss, summ_test_acc = sess.run([preds,total_loss,accuracy,summ_loss_val,summ_acc_val], feed_dict={input_: X_test, batch_labels: Y_test})
            
			#compute custom metrics
            test_task_acc = TaskAccuracy(test_preds, Y_test)
            summ_test_task_acc = sess.run(summ_task_acc_val, feed_dict={task_accuracy: test_task_acc})
            
            test_timing_error = TimingError(test_preds, Y_test)
            summ_test_time_error = sess.run(summ_time_error_val, feed_dict={timing_error: test_timing_error})
            
            writer.add_summary(summ_test_loss, epoch)
            writer.add_summary(summ_test_acc, epoch)
            writer.add_summary(summ_test_task_acc, epoch)
            writer.add_summary(summ_test_time_error, epoch)
            print(" after " + str(epoch+1) + " epochs: test_loss = " + "%.1f"%(test_loss) + ", test_acc = " + "%.2f"%(test_acc) + ", test_task_dtw = " + "%.2f"%(test_task_acc) + ", test_time_dtw = " + "%.2f"%(test_timing_error))
            sys.stdout.flush()



