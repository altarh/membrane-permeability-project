import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from rdkit import Chem
import numpy as np


df = pd.read_csv('CycPeptMPDB_First30.csv', sep=",")
threshold = -6.0
# If Permeability is greater than -6, it's 1 (Active), otherwise 0
df['Class_Label'] = df['Permeability'].apply(lambda x: 1 if x > threshold else 0)

# Check the results
print(df[['Permeability', 'Class_Label']].head(30))

