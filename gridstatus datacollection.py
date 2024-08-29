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

load_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/Load_Data.csv'
load_df.to_csv(load_file_path, index=False)



start = pd.Timestamp("Jan 1, 2023").normalize()
end = pd.Timestamp.now().normalize()
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

lmp_df = caiso.get_lmp(
    start=start, end=end, market="DAY_AHEAD_HOURLY", locations=locations, sleep=5
)

lmp_df['Time'] = lmp_df['Time'].dt.tz_localize(None)

lmp_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/2023_Lmp_Data.csv'
lmp_df.to_csv(lmp_file_path, index=False)



start = pd.Timestamp("Jan 1, 2022").normalize()
end = pd.Timestamp('Dec 31, 2022').normalize()
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

lmp_df = caiso.get_lmp(
    start=start, end=end, market="DAY_AHEAD_HOURLY", locations=locations, sleep=5
)

lmp_df['Time'] = lmp_df['Time'].dt.tz_localize(None)

lmp_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/2022_Lmp_Data.csv'
lmp_df.to_csv(lmp_file_path, index=False)



start = pd.Timestamp("Jan 1, 2021").normalize()
end = pd.Timestamp('Dec 31, 2021').normalize()
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

lmp_df = caiso.get_lmp(
    start=start, end=end, market="DAY_AHEAD_HOURLY", locations=locations, sleep=5
)

lmp_df['Time'] = lmp_df['Time'].dt.tz_localize(None)

lmp_file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Load Data/2021_Lmp_Data.csv'
lmp_df.to_csv(lmp_file_path, index=False)



curtailment_df = caiso.get_curtailment(
    start="Jan 1, 2021", end='Oct 31, 2023')

curtailment_df.shape


