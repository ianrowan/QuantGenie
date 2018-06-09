import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as pdr
from datetime import datetime

class BuildDataset:

    def __init__(self, symbol, years, scale=False):
        self.end = datetime.now().date()
        self.start = datetime(self.end.year-int(years),self.end.month,self.end.day)
        self.data = pdr.DataReader(symbol.upper(),'morningstar', start=self.start, end=self.end).as_matrix()
        print("Stock data received")
        self.data_length = len(self.data)
        for i in range(self.data_length):
            for j in range(len(self.data[0])):
                if self.data[i][j] !=self.data[i][j]:
                    self.data[i][j] = (self.data[i][2] + self.data[i][1])/2
        self.preserve = self.data
        self.scale =scale
        if scale:
            sc = MinMaxScaler()
            self.data = sc.fit_transform(self.data)
            self.min = sc.data_min_
            self.max = sc.data_max_

        self.input_ = np.zeros(shape=[len(self.data)-35, 30, 5])
        self.out = np.zeros(shape=[len(self.data)-35, 5, 5])

    def build(self):
        i = 0
        while i+35< len(self.data):
            self.input_[i] = self.data[i:i+30]
            self.out[i] = self.data[i+30:i+35]
            i += 1
        return self.input_, self.out

    def rescale(self, array):
        return array *(self.max-self.min)+self.min

    def get_time(self):
        return datetime.now().replace(second=0, microsecond=0)

    def get_index(self, index):
        index_full = pdr.DataReader(index.upper(),'morningstar', start=self.start, end=self.end).as_matrix()
        for i in range(len(index_full)):
            for j in range(len(index_full[0])):
                if index_full[i][j] !=index_full[i][j]:
                    index_full[i][j] = (index_full[i][2] + index_full[i][1])/2

        print('Index data received')
        index = index_full[len(index_full)-self.data_length:]

        if self.scale:
            sc = MinMaxScaler()
            index = sc.fit_transform(index)
        input_ = np.zeros(shape=[len(self.data) - 35, 30, 5])
        i = 0
        while i + 35 < len(self.data):
            self.input_[i] = index[i:i + 30]
            i += 1
        return input_, index

    def build_full(self, benchmark):
        index_chan = self.get_index(benchmark)[0]
        stock_chan, out = self.build()
        inpt = np.array([stock_chan,index_chan])
        input_ = np.zeros([len(self.data)-35,30, 5, 2])
        input_[:, :,:,0] = inpt[0,:,:,:]
        input_[:,:,:,1]= inpt[1, :, :,:]
        return input_, out

    def get_recent(self,benchmark):
        stock_chan = self.data[len(self.data)-30:]
        index_chan = self.get_index(benchmark)[1][len(self.data)-30:]
        recent_input = np.array([stock_chan, index_chan])
        return recent_input, self.preserve[len(self.data)-30:]

print(datetime.now().replace(second=0, microsecond=0))