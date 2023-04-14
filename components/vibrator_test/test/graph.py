#!/usr/bin/env python3
import pandas
import sys, numpy
import scipy.fftpack
import matplotlib.pyplot as plt
if len(sys.argv)==2:
    file = pandas.read_csv(sys.argv[1], delimiter=";")
    print(file)
    file = file.drop("tiempo",axis=1)
    #file = file.drop("Unnamed: 0",axis=1)


    '''
    ##Seno sintetico para validar transformada de fourier
    start_time = 0
    end_time = 1
    sample_rate = 1000
    time = numpy.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    frequency = 25
    amplitude = 1
    sinewave = amplitude * numpy.sin(2 * numpy.pi * frequency * time + theta)
    plt.show() 

    signal = sinewave
    fourierTransform = numpy.fft.fft(signal)/len(signal)           # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(signal)/2))]  # Exclude sampling frequency
    tpCount = len(signal)
    values = numpy.arange(int(tpCount/2))
    timePeriod = 1
    frequencies = values/timePeriod
    # Create subplot
    plt.plot(frequencies[1:], abs(fourierTransform[1:]))

    plt.plot(sinewave)
    plt.show()
    '''

    plt.title(sys.argv[1])
    figure, axis = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=1)

    for i in range(3):
        signal = numpy.array(file.values).transpose()[i]
        fourierTransform = numpy.fft.fft(signal)/len(signal)           # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(signal)/2))]  # Exclude sampling frequency
        tpCount = len(signal)
        values = numpy.arange(int(tpCount/2))
        timePeriod = 4
        frequencies = values/timePeriod
        # Create subplot
        axis[i].plot(frequencies[1:], abs(fourierTransform[1:]))
    plt.show()

    
    

    plt.title(sys.argv[1])

    print(file)
    file.plot (subplots=True, kind="line", logy=False, grid=True)
    
    plt.show()

    
else:
    print("Falta la dirección del csv")