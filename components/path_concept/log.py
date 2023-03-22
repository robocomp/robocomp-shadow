# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def main():

    k = np.log(10000)/100
    x = np.arange(0, 100)
    plt.plot(x, np.exp(x*k))
    plt.show()
    print(np.exp(100*k))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


