import os
import warnings
warnings.filterwarnings("ignore")
from StockFCN import train_network
import numpy as np
from Predict import Predict
import warnings
from tkinter import filedialog

def get_params():
    years = input("how many years of data would you like to train on? ")
    steps = input("how many training steps? ")
    return years, steps

def main():
    #np.set_printoptions(precision=2)
    print("Welcome to QuantGenie!")
    stock = input("Input stock symbol you would like to predict: ")
    path = filedialog.askdirectory(initialdir="/", title="Select Data Directory")
    try:
        direc = path+"/"+stock.lower()
    except TypeError:
        direc = "/home/ian/Quant/" + stock.lower()
    print("Directory: {}".format(direc))

    if not os.path.exists(direc):
        input("QuantGenie has not trained a network for " + stock.upper()+" pres ENTER to train one now")
        years, steps = get_params()
        input("Press ENTER to Commence training")
        train_network(stock, years, steps, direc)

    timestamp = np.genfromtxt(direc+"/time.txt", dtype=np.str)
    print("Last Training of {} occured at {}. ".format(stock, timestamp))
    retrain = input("Select an option: \n(0) Predict 5 days\n(1) Retrain")

    if retrain=='1':
        years, steps = get_params()
        pre = Predict(stock, years, direc)
        input("Press ENTER to Commence training")
        pre.retrain(steps)
        print("Training Complete, Predicting.......")
        prediction = pre.predict()
        pre.plot_chart(prediction)
        pre.write_file(prediction)
    elif retrain == '0':
        pre = Predict(stock, 17,direc)
        prediction = pre.predict()
        pre.plot_chart(prediction)
        pre.write_file(prediction)
    print("Job, Complete! Run again for a new stock")

if __name__== "__main__":
    warnings.filterwarnings('ignore')
    main()