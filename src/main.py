#lib
import pandas as pd

#def
import processing_data as prodas

#main
def main():
    data_path = "..\\data\\iris.csv"
    data = pd.read_csv(data_path)
    prodas.proc(data) #data processing

if __name__ == "__main__":
    main()
