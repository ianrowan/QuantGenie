import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from BuildDataset import BuildDataset
import pandas as pd
from tqdm import tqdm_gui

class Predict(BuildDataset):

    def __init__(self, symbol, years, direct, scale=True):
        BuildDataset.__init__(self,symbol=symbol, years=years, scale=scale)
        self.symbol = symbol
        self.direc = direct

    def predict(self):
        meta = '{}/{}_model.ckpt.meta'.format(self.direc, self.symbol.lower())
        ckpt = '{}/{}_model.ckpt'.format(self.direc, self.symbol.lower())
        recent, stock = self.get_recent('QQQ')
        inp = np.zeros([1,30,5,1])
        inp[0, :,:, 0] = recent[0, :,:]
        #inp[0,:,:, 1] = recent[1,:,:]
        print('Feeding Data...')
        print(pd.DataFrame(stock, columns=['Close', 'High', 'Low', 'Open', 'Volume']))
        print("Predicting.......")
        with tf.Session() as sess:
            imp = tf.train.import_meta_graph(meta)
            imp.restore(sess, ckpt)
            pred = sess.run("Up_conv3/dconv2/Relu:0", feed_dict={'input_:0': inp})

        return self.rescale(np.reshape(pred[0], [5,5]))

    def retrain(self, steps):
        inp, out = self.build_full('QQQ')
        inp = np.reshape(inp[:, :, :, 0], newshape=[-1, 30, 5, 1])
        out = np.reshape(out, [-1, 5, 5, 1])
        meta = '{}/{}_model.ckpt.meta'.format(self.direc, self.symbol.lower())
        ckpt = '{}/{}_model.ckpt'.format(self.direc, self.symbol.lower())
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta)
            saver.restore(sess, ckpt)
            for i in tqdm_gui(range(int(steps))):

                if i % 50 == 0:
                    train_accuracy = sess.run("Mean:0", feed_dict={"input_:0":inp, "exp_out:0":out})
                    ac = sess.run("Sub:0", feed_dict={"input_:0": inp, "exp_out:0": out})
                    print('========================SUMMARY REPORT=============================')
                    print('step %d, train loss: %g' % (i, train_accuracy))
                    print('Validation accuracy {}%'.format(str(ac)))
                    # print('Estimated Time Remaining = ' + str(round((20000-i)*(timer/60)/60,2)) + ' Hours')
                    print('===================================================================')
                sess.run("Adam", feed_dict={"input_:0":inp, "exp_out:0":out})
            saver.save(sess, ckpt)
            with open(self.direc + "/time.txt", 'w') as f:
                f.write('%s' % (self.get_time()))
            print(self.get_time())


    def write_file(self, pred):
        np.savetxt('{}/5_pred'.format(self.direc), pred, delimiter=',')
        print('Data Written to {}/5_pred'.format(self.direc))

    def weekdays(self):
        dates = []
        if datetime.now().time() < datetime(hour=16, year=2018, month=5, day=1).time():
            start_date = datetime.now()
        else:
            start_date = datetime.now() + timedelta(days=1)

        if start_date.weekday() == 5:
            start_date = start_date + timedelta(days=2)
        elif start_date.weekday() == 6:
            start_date= start_date + timedelta(days=1)
        last = start_date
        dates.append(start_date.date())
        for i in range(4):
            current = last + timedelta(days=1)
            if current.weekday() == 5:
                current = current + timedelta(days=2)
            elif current.weekday() == 6:
                current = current + timedelta(days=1)
            dates.append(current.date())
            last = current
        return dates

    def plot_chart(self, pred, show=True):
        if show:
            fig = plt.figure()
        ax1 = plt.subplot2grid((1,1),(0, 0))
        close, high, low, openp, volume = pred.transpose()
        dates = mdates.date2num(self.weekdays())
        x = pd.DataFrame(pred, index=self.weekdays(), columns=['Close', 'High', 'Low', 'Open', 'Volume'],)
        print(x)
        ohlc = []
        for i in range(len(dates)):
            append_me = dates[i], openp[i], high[i], low[i], close[i], volume[i]
            ohlc.append(append_me)
        candlestick_ohlc(ax1, ohlc, width=0.5, colorup='#77d879', colordown='#db3f3f')
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(8))
        ax1.grid(True,axis='y')
        plt.xlabel('Date')
        plt.ylabel('Price($)')
        plt.title(self.symbol + " 5 day Prediction")
        plt.legend()
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
        plt.savefig(self.direc+"/{}_chart".format(self.symbol))
        if show:
            plt.show()
            print


