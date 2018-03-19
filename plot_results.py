#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:41:10 2018

@author: user
"""
import matplotlib.pyplot as plt



def plot_loss(name, experimentPath, num_iters,train_err, test_err):
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.title('RNN\n~~~ train & test errors for %s ~~~\n' %(name))
    plt.semilogy(range(num_iters),train_err , 'r', label="Train Error")
    plt.semilogy(range(num_iters),test_err , 'b', label="Test Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(experimentPath+name+'.png')
    plt.clf()


def plot_hist(np_array, mytitle):
    plt.hist(np_array,bins=5)
    plt.title('Histogram of STFT values - ' + mytitle)
    plt.show()
    plt.clf()

