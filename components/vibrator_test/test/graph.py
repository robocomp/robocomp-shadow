#!/usr/bin/env python3
import pandas
import sys, numpy
import matplotlib.pyplot as plt

def get_mov(t):
    movimiento = {"Avance":10, "Giro": 16.25, "Lateral": 26.25, "Giro2":27.5, "Parada1":27.75,
                "Avance Rapido":31.5, "Parada2":31.75, "Giro3":38, "Parada3":38.25, "Lateral Rapido": 42
                , "Parada4":42.25, "Giro4": 43.25, "Diagonal":48.25, "Parada5":48.5, "Diagonal Rapido":52.25, 
                "Giro5":53.5}
    for key, value in movimiento.items():
        if t<value:
            return key
    return "Out Range"

def  test_fourier():
    ##Seno sintetico para validar transformada de fourier
    start_time = 0
    end_time = 1
    sample_rate = 1000
    time = numpy.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    frequency = 250
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



####################################################MAIN###############################################
if len(sys.argv)==3:
    #Cargamos los datos
    file = pandas.read_csv(sys.argv[1], delimiter=";")
    print(file)

    #Eliminamos la columna de tiempo
    file = file.drop("tiempo",axis=1)
    #file = file.drop("Unnamed: 0",axis=1)
    
    #Plot data
    file.plot (subplots=True, kind="line", logy=False, grid=True)
    plt.draw()

    #Array con los datos x, y, z
    signalFull = numpy.array(file.values).transpose()
    frecuenciaMuestreo = 1000//float(sys.argv[2])
    tVentana = 1 #tiempo de ventana en segundos
    windowsSize = int(tVentana*frecuenciaMuestreo)
    values = numpy.arange(int(windowsSize/2))
    values = values * frecuenciaMuestreo / windowsSize
    timePeriod = float(sys.argv[2])
    frequencies = values


    ### Espectograma
    espectogram, eAxis = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.5)
    
    for i in range(3):
       eAxis[i].specgram(x=signalFull[i], Fs=frecuenciaMuestreo)
       #eAxis[i].specgram(x=signalFull[i], Fs=frecuenciaMuestreo, window=windowsSize, NFFT=windowsSize)
       eAxis[i].set_ylabel("Hz", fontsize=10)
       eAxis[i].set_xlabel("Time second", fontsize=10)
    plt.draw()


    ##### Fourier manual
    figure, axis = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.75)
    overLap = windowsSize//2
    end = len(signalFull[0]) - windowsSize
    for t in range(0,end, overLap):
        for i in range(3):
            signal =  signalFull[i][t:t+windowsSize]
            fourierTransform = numpy.fft.fft(signal)/len(signal)           # Normalize amplitude
            fourierTransform = fourierTransform[range(int(len(signal)/2))]  # Exclude sampling frequency
            # Create subplot
            axis[i].cla() 
            axis[i].set_ylim(0, 3)
            axis[i].set_title("Movimiento "+get_mov(t*timePeriod/1000)+"/ /Ventana "+str(t*timePeriod/1000)+" s a "+str((t+windowsSize)*timePeriod/1000)+" s", fontsize=10)
            axis[i].set_xlabel("Hz", fontsize=10)
            axis[i].set_xticks(numpy.arange(0, frequencies[-1], 10))
            axis[i].set_yticks(numpy.arange(0, 3, 0.5))
            axis[i].set_ylabel("Acceleration", fontsize=10)
            axis[i].grid()
            axis[i].plot(frequencies[1:], abs(fourierTransform[1:]))
        plt.draw()
        plt.pause(0.25)
else:
    print("Falta la dirección del csv o periodo de muestreo en ms")