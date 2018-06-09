import numpy as np
import tqdm as tqdm
import tensorflow as tf
from time import time
from sklearn.model_selection import train_test_split
from BuildDataset import BuildDataset
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm_gui
import csv
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

def train_network(stock, years, steps, direct, scale=True):

    bd = BuildDataset(symbol=stock, years=years, scale=scale)
    inp, out = bd.build_full('QQQ')

    inp=np.reshape(inp[:,:,:,0], newshape=[-1, 30,5,1])

    out = np.reshape(out, [-1, 5,5,1])
    print("Input data size:" + str(inp.shape))
    print("Output data size:" + str(out.shape))

    inp, inp_v, y, y_v = train_test_split(inp, out, test_size=.10)

    print(inp_v.shape)
    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def weight_var(shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def bias_var(shape):
        init = tf.constant(0.0, shape=shape)
        return tf.Variable(init)

    def conv2d (x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def Conv_Layer(input_, shape, name, str=1, pool=False):
        with tf.variable_scope(name):
            with tf. name_scope('Weights'):
                w = weight_var(shape)
            with tf.name_scope('Bias'):
                b = bias_var([shape[3]])

            h_conv = tf.nn.relu(conv2d(input_, w, stride=str)+ b)

            if pool == True:
                return max_pool_2x2(h_conv)
        return h_conv
    with tf.device("/gpu:0"):
        x_ = tf.placeholder(tf.float32, shape=[None, 30, 5, 1], name='input_')
        y_exp = tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='exp_out')

        #Conv 1
        cnv1 = Conv_Layer(x_, [3, 3, 1, 6], name="conv1")

        #Conv 2
        cnv2 = Conv_Layer(cnv1, [3, 3, 6, 6], name="conv2")
        with tf.name_scope("max_pool"):
            cnv2 = max_pool_2x2(cnv2)

        cnv22 = Conv_Layer(cnv2, [3, 3, 6, 16], name="conv22")
        cnv23 = Conv_Layer(cnv22, [3, 3, 16, 32], name="conv23")
        #Conv 3
        cnv3 = Conv_Layer(cnv23, [3, 3, 32, 32], name="conv3")

        #Conv 4
        cnv4 = Conv_Layer(cnv3, [3, 3, 32, 32], name="conv4")
        with tf.name_scope("max_pool"):
            cnv4 = max_pool_2x2(cnv4)

        #Fully connected layers

        with tf.name_scope("fully_connected1"):
            w_1 = weight_var([8*2*32, 1000])
            b1 = bias_var([1000])
            c4_flat = tf.reshape(cnv4, [-1, 8*2*32])
            fc1 = tf.nn.relu(tf.matmul(c4_flat, w_1) + b1)

        with tf.name_scope("fully_connected2"):
            w_2 = weight_var([1000, 800])
            b2 = bias_var([800])
            fc2 = tf.nn.relu(tf.matmul(fc1, w_2) + b2)

        with tf.name_scope("fully_connected3"):
            w_3 = weight_var([800, 400])
            b3 = bias_var([400])
            fc3 = tf.nn.relu(tf.matmul(fc2, w_3) + b3)

        with tf.name_scope("Up_conv1"):
            fc3 = tf.reshape(fc3, shape=[-1, 20, 20, 1])
            #d_cnv1 = tf.layers.conv2d_transpose(fc3, filters=16 , kernel_size = 2, strides=2)
            cnvu1 = Conv_Layer(fc3, shape=[3, 3, 1, 6], name="dconv1", pool=True)

        with tf.name_scope("Up_conv2"):
            cnvu2 = Conv_Layer(cnvu1, shape=[3, 3, 6, 16], name="dconv1", pool=False)
            cnvu2 = Conv_Layer(cnvu2, shape=[3, 3, 16, 6], name="dconv2", pool=True)

        with tf.name_scope("Up_conv3"):
            #d_cnv2 = tf.layers.conv2d_transpose(cnvu1, filters=6, kernel_size=1, strides=1)
            logits = Conv_Layer(cnvu2, shape=[3, 3, 6, 1], name="dconv2")
            tf.identity(logits, name="y_out")

        #real_c = tf.summary.image("Exp", y_exp, max_outputs=2)
        #pred_c = tf.summary.image("pred", logits, max_outputs=2)

        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_exp, predictions=logits))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        acc = tf.reduce_mean(tf.losses.absolute_difference(labels=y_exp,predictions=logits))
        acc = tf.cast(tf.subtract(tf.constant(100, dtype=tf.float32),tf.multiply(acc, tf.constant(100, dtype=tf.float32))), tf.float32)
        train_c = tf.summary.scalar("Train_loss", loss)
        val_c = tf.summary.scalar('Val_loss', loss)
        Accuracy = tf.cast(loss, tf.float32)
        print(acc)
        print(Accuracy)

        saver = tf.train.Saver(tf.all_variables())
        # TF SESSION
        config =tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth =True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            timer = 0
            train_start =time()
            write = tf.summary.FileWriter('/home/ian/Quant', sess.graph)
            j= 0
            loss = 0
            patience = 0
            for i in tqdm_gui(range(int(steps))):
                time_start = time()
                #batch = next_batch(len(), inp, y)

                if i% 5==0:
                    lo = acc.eval(feed_dict={x_: inp_v, y_exp: y_v})
                    if lo-loss < .0005:
                        patience +=1
                    else:
                        patience=0
                    loss = lo
                    #print(patience)
                if patience == 5:
                    break

                if i % 50 == 0:

                    train_accuracy = Accuracy.eval(feed_dict={x_: inp, y_exp: y})
                    ac = acc.eval(feed_dict={x_: inp_v, y_exp: y_v})
                    tc = train_c.eval(feed_dict={x_: inp, y_exp: y})
                    vc = val_c.eval(feed_dict={x_: inp_v, y_exp: y_v})
                    #image1, image2 = sess.run([real_c, pred_c],feed_dict={x_: batch[0][:10], y_exp: batch[1][:10]})
                    #p = int(100 * np.random.rand())
                    #print(bd.rescale(logits.eval(feed_dict={x_: inp_v[:100]})[p]).astype(np.int))
                    #print(bd.rescale(y_v[p]).astype(np.int))
                    #write.add_summary(image1, i)
                    #write.add_summary(image2, i)
                    write.add_summary(tc, i)
                    write.add_summary(vc, i)

                    print('========================SUMMARY REPORT=============================')
                    print('step %d, train loss: %g' % (i,train_accuracy))
                    print('Validation accuracy {}%'.format(str(ac)))
                    #print('Estimated Time Remaining = ' + str(round((20000-i)*(timer/60)/60,2)) + ' Hours')
                    print('===================================================================')

                train_step.run(feed_dict={x_: inp, y_exp: y})
                time_stop = time()
                timer = time_stop - time_start

                #print('Step: ' + str(i) + ' Epoch Time: ' + str(round(timer,2)) + ' Secs.' + ' Time elapsed: ' +
                      #str(round((time_stop-train_start)/60, 2)) + ' Mins.' + str(round((i/80000) *100, 1)) + " % complete")

            directory = direct
            if not os.path.exists(directory):
                os.makedirs(directory)

            s_path = saver.save(sess, "{}/{}_model.ckpt".format(directory, stock.lower()))
            print("model saved in {}".format(s_path))

            with open(directory+"/time.txt", 'w') as f:
                f.write('%s' % (bd.get_time()))
            print(bd.get_time())
