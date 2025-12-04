import pandas as pd
from pathlib import Path
import re

class DataHandler:
    def __init__(self, data):
        self.data = data    
    def getData(self):
        file = Path(self.data)
        if not file.exists():
            return print("A fájl nem létezik!");
        if(self.data.endswith('.txt') or self.data.endswith('.csv')):
            df = pd.read_csv(self.data, sep="\s+|;|:|,|\t+", engine="python")
            return df["Xmax"].to_numpy()
        else: 
            return print("Nem megfelelő a fájl kiterjesztése!");
        


