import os
import warnings
warnings.filterwarnings("ignore")
from StockFCN_t import train_network1
from Predict_t import Predict
from tkinter import filedialog
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time


class Portfolio:

    def __init__(self, time, directory, port_file):
        self.date = time.day
        self.time = time.time()
        self.directory_base = directory
        self.stocks = self.get_port(port_file)

    def get_port(self, file):
        return np.genfromtxt(file, dtype=np.str, delimiter=',')

    def train_networks(self):
        for i in tqdm(self.stocks):
            tqdm.write("Checking {} Network".format(i.upper()))
            direc = self.directory_base+'/{}'.format(i.lower())
            if not os.path.exists(direc):
                train_network1(i, 18, 7000, direc)
                pre = Predict(i, 18, direc)
                predict = pre.predict()
                pre.plot_chart(predict, show=False)
            else:
                last_train = np.genfromtxt(direc + "/time.txt", dtype=np.str, delimiter=' ')
                #print(last_train)
                if (int(last_train[0][8:])<self.date and self.time.hour>=16 ) or (int(last_train[1][:2])<16 and int(last_train[0][8:]) != self.date):
                    pre = Predict(i, 18, direc)
                    pre.retrain(400)
                    predict = pre.predict()
                    pre.plot_chart(predict,show=False)
                else:
                    tqdm.write("{} is up to date".format(i.upper()))

def main():
    #stocks=['aapl','amzn','ayx','clf','fas','FB','nvda','pran','swch']
    directory = '/home/ian/Portfolio'
    stocks = '/home/ian/Dropbox/Portfolio/portfolio.txt'
    while True:
        time_now = datetime.now()
        print(time_now)
        Portfolio(time_now, directory, port_file=stocks).train_networks()
        time.sleep(10*60)



if __name__ == '__main__':
    main()








