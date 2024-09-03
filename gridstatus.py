# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 03:39:39 2023

@author: dashi
"""

# import isodata
import gridstatus
import pandas as pd
import xlsxwriter
import pytz
import plotly.express as px


caiso = gridstatus.CAISO()

df = pd.DataFrame()
df = caiso.get_lmp_today('DAY_AHEAD_HOURLY', locations=["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"])
df.shape
df.columns


start = pd.Timestamp("January 1, 2021").normalize()
end = pd.Timestamp.now().normalize()
load_df = caiso.get_load(start, end=end)

load_df['Time'] = load_df['Time'].dt.tz_localize(None)

load_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Code/Load Data/Load_Data.csv'
load_df.to_csv(load_file_path, index=False)



start = pd.Timestamp("Jan 1, 2021").normalize()
end = pd.Timestamp('Dec 31, 2021').normalize() 
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

lmp_df = caiso.get_lmp(
    start=start, end=end, market="DAY_AHEAD_HOURLY", locations=locations, sleep=5
)

lmp_df['Time'] = lmp_df['Time'].dt.tz_localize(None)

lmp_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/LMP Data/2021_Lmp_Data'
lmp_df.to_csv(lmp_file_path, index=False)





start = pd.Timestamp("Nov 1, 2023").normalize()
end = pd.Timestamp('Nov 29, 2023').normalize() 
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

lmp_df = caiso.get_lmp(
    start=start, end=end, market="DAY_AHEAD_HOURLY", locations=locations, sleep=5
)

lmp_df['Time'] = lmp_df['Time'].dt.tz_localize(None)

lmp_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/LMP Data/Nov_23_Lmp_Data.csv'
lmp_df.to_csv(lmp_file_path, index=False)



df2022 = pd.read_csv('C:/Users\dashi\OneDrive\Desktop\CAISO Projects\Data\LMP Data/2022_Lmp_Data.csv')
df2023 = pd.read_csv('C:/Users\dashi\OneDrive\Desktop\CAISO Projects\Data\LMP Data/2023_Lmp_Data.csv')

comb_df = pd.concat([df2022, df2023], ignore_index=True)
comb_df.columns

comb_df['Time'] = pd.to_datetime(comb_df['Time'])
comb_df['Time'] = comb_df['Time'].dt.tz_localize(None)
print(comb_df.duplicated().sum())
comb_df = comb_df.drop_duplicates()
new_file_path = 'C:/Users\dashi\OneDrive\Desktop\CAISO Projects\Data\LMP Data/2022&2023_Lmp_data.csv'
comb_df.to_csv(new_file_path, index=False)

comb_df.head(30)
comb_df['Energy']
comb_df['LMP']



start = pd.Timestamp("Nov 1, 2023").normalize()
end = pd.Timestamp.now().normalize()
load_df = caiso.get_load(start, end=end)

load_df['Time'] = load_df['Time'].dt.tz_localize(None)

load_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/Nov_23_Load.csv'
load_df.to_csv(load_file_path, index=False)


old_load_df = pd.read_csv('C:/Users\dashi\OneDrive\Desktop\CAISO Projects\Data\Load Data/Load_Data.csv')
filtered_load_df = old_load_df[old_load_df['Time'] >= '2023-11-01']
file_path = 'C:/Users\dashi\OneDrive\Desktop\CAISO Projects\Data\Load Data/Load_Data.csv'
filtered_load_df.to_csv(file_path, index=False)
