import pandas as pd
from pathlib import Path

class DataHandler:
    def __init__(self, data):
        self.data = data  
    def getData(self):
        file = Path(self.data)
        if not file.exists():
            return print("A fájl nem létezik!");
        if(self.data.endswith('.txt') or self.data.endswith('.csv')):
            df = pd.read_csv(self.data, delim_whitespace=True)
            return df["Xmax"].to_numpy(), df["Counts"].to_numpy(), df["CountsSqrt"].to_numpy()
        else: 
            return print("Nem megfelelő a fájl kiterjesztése!");
        


