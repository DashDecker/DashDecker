# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 02:57:42 2023

@author: dashi
"""

import pandas as pd
import isodata
import xlsxwriter
import pytz


directory = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Code/Historical Demand Data'

caiso = isodata.CAISO()

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

month_data = {}


for month, num_days in zip(months, days_in_month):
    month_data[month] = pd.DataFrame()
    
    for day in range(1, num_days + 1):
        data = caiso.get_historical_demand(f'{month} {day}, 2021')
        month_data[month] = pd.concat([month_data[month], data], ignore_index=True)
        
    month_data[month]['Time'] = month_data[month]['Time'].dt.tz_localize(None)

    writer = pd.ExcelWriter(f'{directory}2021_{month}_data.xlsx', engine='xlsxwriter')
    month_data[month].to_excel(writer, sheet_name=f'2021 {month}', index=False)
    writer.sheets[f'2021 {month}'].set_column('A:A', 20)
    writer.close()



for month, num_days in zip(months, days_in_month):
    month_data[month] = pd.DataFrame()
    
    for day in range(1, num_days + 1):
        data = caiso.get_historical_demand(f'{month} {day}, 2022')
        month_data[month] = pd.concat([month_data[month], data], ignore_index=True)
        
    month_data[month]['Time'] = month_data[month]['Time'].dt.tz_localize(None)

    writer = pd.ExcelWriter(f'{directory}2022_{month}_data.xlsx', engine='xlsxwriter')
    month_data[month].to_excel(writer, sheet_name=f'2022 {month}', index=False)
    writer.sheets[f'2022 {month}'].set_column('A:A', 20)
    writer.close()
    
    

for month, num_days in zip(months, days_in_month):
    if month in ('Nov', 'Dec'):
        continue
    
    month_data[month] = pd.DataFrame()
    
    for day in range(1, num_days + 1):
        data = caiso.get_historical_demand(f'{month} {day}, 2023')
        month_data[month] = pd.concat([month_data[month], data], ignore_index=True)
        
    month_data[month]['Time'] = month_data[month]['Time'].dt.tz_localize(None)

    writer = pd.ExcelWriter(f'{directory}2023_{month}_data.xlsx', engine='xlsxwriter')
    month_data[month].to_excel(writer, sheet_name=f'2023 {month}', index=False)
    writer.sheets[f'2023 {month}'].set_column('A:A', 20)
    writer.close()
    
    
    
    
    
