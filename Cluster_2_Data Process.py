import pandas as pd
import numpy as np

df = pd.read_csv('In_Data_2.csv', encoding = "ISO-8859-1")

exclude=['BALANCED SALT SOLUTION - UNSPECIFIED %',
         'CHLORHEXIDINE TOPICAL 0.12%',
         'CHLORHEXIDINE',
         'DEXTRAN 40',
         'DEXTROSE',
         'DEXTROSE / LACTATED RINGERS 5%',
         'DEXTROSE / SALINE 5% / 0.225%',
         'DEXTROSE / SALINE 5% / 0.45%',
         'DEXTROSE / SALINE 5% / 0.9%',
         'DEXTROSE / SALINE W/KCL 5% / 0.45% + 10 MEQ/L',
         'DEXTROSE / SALINE W/KCL 5% / 0.45% + 20 MEQ/L',
         'DEXTROSE / SALINE W/KCL 5% / 0.9% + 20 MEQ/L',
         'DEXTROSE / WATER 10%',
         'DEXTROSE / WATER 5%',
         'DEXTROSE 50%',
         'DOCUSATE/SENNA 50/8.6',
         'ELECTROLYTE-A SOLUTION',
         'General - Medications - Current',
         'General - Medications - Home',
         'INHALATIONAL',
         'Mean Inspiratory Pressure',
         'OTHER - EPIDURAL MEDICATION',
         'OTHER - INTRATHECAL MEDICATION',
         'OTHER - INTRAVENOUS MEDICATION',
         'OTHER - LOCAL INFILTRATION MEDICATION',
         'OTHER - ORAL MEDICATION',
         'OTHER - UNSPECIFIED ROUTE MEDICATION',
         'OTHER FLUID IN, MEDICATION INFUSION VOLUME',
         'OTHER FLUID IN, UNSPECIFIED',
         'SALINE 0.45%',
         'SALINE 0.9%',
         'SALINE 5%',
         'SODIUM CHLORIDE 0.9%',
         'SODIUM CITRATE',
         'Uncategorized \x96 Preop/PACU/Floor/ICU nursing documentation',
         'Unknown Concept'
]

df = df[~df['concept_desc'].isin(exclude)]
df = df[df.groupby("concept_desc").concept_desc.transform("size") >= 5]
df = df.drop(columns=['PROCEDURE_TXT'])

print(len(set(df['concept_desc'])))
print(len(set(df['MPOG_CASE_ID'])))
print(df.shape)

print(set(df['concept_desc']))

df.replace('ACETYLCYSTEINE 20%', 'ACETYLCYSTEINE')
df.replace(['ALBUMIN 25%', 'ALBUMIN 5%'], 'ALBUMIN')
df.replace('ALBUTEROL / IPRATROPIUM  2.5MG / 0.5MG', 'ALBUTEROL')
df.replace('ATROPINE 1%', 'ATROPINE')

cases = list(set(df['MPOG_CASE_ID']))
meds = list(set(df['concept_desc']))
df2 = pd.DataFrame(0, index=cases, columns=meds)
for index, row in df.iterrows():
    df2.at[row['MPOG_CASE_ID'], row['concept_desc']] = 1


print(df2.values.sum())
df2.to_csv('Mid_Data_2_full.csv')
