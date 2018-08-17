import pandas as pd
import numpy as np

df = pd.read_csv('In_Data_1.csv')
df.drop(columns=['ArterialLine_ArterialLine', 'NerveBlock_PeripheralNerveBlock', 'NerveBlock_NeuraxialNerveBlock',\
                'AnesTechnique_MAC', 'AnesTechnique_General', 'Medications_InhaledAnesthetic', 'Medications_IVSedativeHypnotic', \
                'Medications_Benzodiazepine', 'Medications_Opioid', 'Group_All_Values', 'AIMS_Scheduled_DT', 'MPOG_Institution_ID', 'PROCEDURE_TXT'], inplace=True)
print(df.shape)

#female: 1, male: 0
def assign_sex(x):
    if x == 'Male':
        return 0
    elif x == 'Female':
        return 1
    else:
        return None

df['Sex'] = df['Sex'].apply(assign_sex)

df = df[df['ASA'] != -999.0]
df = df[df['ASA'] != -998.0]
df.dropna(inplace=True)
df['ASA'] = (df.ASA.max() - df['ASA']) / (df.ASA.max() - df.ASA.min())
df['Age'] = (df.Age.max() - df['Age']) / (df.Age.max() - df.Age.min())
df['Sex'] = df['Sex'].astype(int)

print(df.shape)
df.to_csv('Mid_Data_1_full.csv')
df = df.drop(columns=['MPOG_CASE_ID','CPT', 'CPT_Predicted'])
df.to_csv('Mid_Data_1_clean.csv')