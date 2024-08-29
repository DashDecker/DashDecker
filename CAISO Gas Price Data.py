# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:41:52 2023

@author: dashi
"""

import gridstatus
import pandas as pd

caiso = gridstatus.CAISO()

start = start = pd.Timestamp("Jan 01, 2021").normalize()
end = pd.Timestamp.now().normalize()

gas_price_df = caiso.get_gas_prices(start=start, end=end, fuel_region_id="ALL")

gas_price_df['Time'] = gas_price_df['Time'].dt.tz_localize(None)

file_path = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Data/Misc Data/Gas_price_data.csv'
gas_price_df.to_csv(file_path, index=False)


