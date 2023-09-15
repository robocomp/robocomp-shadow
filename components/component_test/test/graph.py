#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def get_movement(time):
    movements = {"Forward": 10, "Turn": 16.25, "Sideways": 26.25, "Turn2": 27.5, "Stop1": 27.75,
                 "Fast Forward": 31.5, "Stop2": 31.75, "Turn3": 38, "Stop3": 38.25, "Fast Sideways": 42,
                 "Stop4": 42.25, "Turn4": 43.25, "Diagonal": 48.25, "Stop5": 48.5, "Fast Diagonal": 52.25,
                 "Turn5": 53.5}
    for key, value in movements.items():
        if time < value:
            return key
    return "Out of Range"

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


def plot_signal(title, signal, time_period, frequencies, windows_size, frecuencia_muestreo):
    ### Spectrogram
    _, eAxis = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.5)

    for i, coord in enumerate(["x", "y", "z"]):
        eAxis[i].specgram(x=signal[i], Fs=frecuencia_muestreo)
        eAxis[i].set_ylabel("Hz", fontsize=10)
        eAxis[i].set_xlabel("Time second", fontsize=10)
        eAxis[i].set_title("Axis "+coord)
    plt.suptitle("Spectrogram of " + title , fontsize=16)
    
    plt.show()

    ##### Manual Fourier
    figure, axis = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.75)
    overlap = windows_size // 2
    end = len(signal[0]) - windows_size
    primero = False
    for t in range(0, end, overlap):
        for i, coord in enumerate(["x", "y", "z"]):
            signal_window = signal[i][t:t + windows_size]
            fourier_transform = np.fft.fft(signal_window) / len(signal_window)
            fourier_transform = fourier_transform[range(int(len(signal_window) / 2))]

            axis[i].cla()
            axis[i].set_ylim(0, 3)
            axis[i].set_title("Movement " + get_movement(t * time_period / 1000) + " / Window " + 
                              str(t * time_period / 1000) + " s to " + str((t + windows_size) * time_period / 1000) +
                                " s / in Axis "+ coord, fontsize=10)
            axis[i].set_xlabel("Hz", fontsize=10)
            axis[i].set_xticks(np.arange(0, frequencies[-1], 10))
            axis[i].set_yticks(np.arange(0, 3, 0.5))
            axis[i].set_ylabel("mm/s^2", fontsize=10)
            axis[i].grid()
            axis[i].plot(frequencies[1:], abs(fourier_transform[1:]))
        plt.suptitle("FFT of " + title, fontsize=16)
        plt.draw()
        if not primero:
            primero = True
            plt.pause(60)
        else:
            plt.pause(0.25)

def process_csv_file(title, file, time_period, max_y, min_y, samples):
    

    # Plot data
    file.plot(xlabel="Time x4 milliseconds", ylabel="mm/s^2", subplots=True, kind="line", logy=False, grid=True, ylim=[min_y,max_y], title="Accelerations of " + title, xlim = [0, samples])
    plt.show()

    if 'ax' in file:
        # Array with x, y, z data
        signal_full = np.array(file.values).transpose()
        frecuencia_muestreo = 1000 // time_period
        t_ventana = 1  # Window time in seconds
        windows_size = int(t_ventana * frecuencia_muestreo)
        values = np.arange(int(windows_size / 2))
        values = values * frecuencia_muestreo / windows_size
        frequencies = values

        plot_signal(title, signal_full, time_period, frequencies, windows_size, frecuencia_muestreo)


def process_csv_files(data_directory, time_period):
    csv_files = [file for file in os.listdir(data_directory) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    Processes = []
    files = {}
    max = None
    min = None
    samples = 0
    for csv_file in csv_files:
        csv_file_path = os.path.join(data_directory, csv_file)
        print(f"Processing file: {csv_file_path}")
        # Load data
        file = pd.read_csv(csv_file_path, delimiter=";")
        print(file)

        # Remove the "time" column
        file = file.drop("tiempo", axis=1)

        if max is None: max = file.max().max()
        elif (aux:=file.max().max())>max: max = aux
        
        if min is None: min = file.min().min() 
        elif (aux:=file.min().min())<min: min = aux
        if (aux:=file.shape[0])>samples: samples = aux

        files[csv_file[0:-4].replace("_", " ").replace("-", "/")] = file
        
    print("Max: ", max, "/Min: ", min, "/Num samples: ", samples)
    # Start all Processes
    for title, file in files.items():
        proc = multiprocessing.Process(target=process_csv_file, args=(title, file, time_period, max, min, samples))
        Processes.append(proc)
        proc.start()

    # Wait for all Processes to complete
    for proc in Processes:
        proc.join()

if len(sys.argv) == 2:
    data_directory = sys.argv[1]
    time_period = float(input("Enter the time period in ms: "))
    process_csv_files(data_directory, time_period)
else:
    print("Please provide the directory path containing the CSV files.")