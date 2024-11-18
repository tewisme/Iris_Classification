#lib
import pandas as pd

#func
import processing_data as prodas

#def
def main():
    data_path = "..\\data\\iris.csv"
    data = pd.read_csv(data_path)
    prodas.proc(data) #data processing

#main
if __name__ == "__main__":
    main()
