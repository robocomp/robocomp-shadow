import pandas
import sys
import matplotlib.pyplot as plt
if len(sys.argv)==2:
    file = pandas.read_csv(sys.argv[1], delimiter=",")
    file = file.drop("tiempo",axis=1)
    file = file.drop("Unnamed: 0",axis=1)
    print(file)
    file.plot (subplots=True, kind="line", logy=False, grid=True)
    plt.show()
else:
    print("Falta la dirección del csv")