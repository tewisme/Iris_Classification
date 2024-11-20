#lib
import pandas as pd

#file
import processing_data as prodas
import algorithms as aas


#func
def main():
    data_path = "..\\data\\iris.csv"
    data = pd.read_csv(data_path)
    #prodas.proc(data) #data processing
    aas.proc(data)

#main
if __name__ == "__main__":
    main()
