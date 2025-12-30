import pandas as pd
import numpy as np
import ast


def read_file_and_add_Class_Label(csv_path='CycPeptMPDB_First30.csv'):
    df = pd.read_csv(csv_path, sep=",")
    threshold = -6.0
    # If Permeability is greater than -6, it's 1 (Active), otherwise 0
    df['Class_Label'] = df['Permeability'].apply(lambda x: 1 if x > threshold else 0)
    return df
