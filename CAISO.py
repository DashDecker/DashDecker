# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:21:41 2023

@author: dashi
"""

import pandas as pd
import isodata
import xlsxwriter
import pytz


directory = 'C:/Users/dashi/OneDrive/Desktop/CAISO Projects/Code/CAISO_Historical_Demand_Data.xlsx'
caiso = isodata.CAISO()

# caiso.get_historical_demand('Sep 1, 2023')

# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

days = list(range(1, 32))

oct_2022 = pd.DataFrame()
nov_2022 = pd.DataFrame()
dec_2022 = pd.DataFrame()
jan_2023 = pd.DataFrame()
feb_2023 = pd.DataFrame()
mar_2023 = pd.DataFrame()
apr_2023 = pd.DataFrame()
may_2023 = pd.DataFrame()
jun_2023 = pd.DataFrame()
jul_2023 = pd.DataFrame()
aug_2023 = pd.DataFrame()
sep_2023 = pd.DataFrame()
oct_2023 = pd.DataFrame()



# for month in months:
    

for day in days:
    oct_22_data = caiso.get_historical_demand(f'Oct {day}, 2022') 
    
    oct_2022 = pd.concat([oct_2022, oct_22_data], ignore_index=True)
    
oct_2022['Time'] = oct_2022['Time'].dt.tz_localize(None)

writer_oct_22 = pd.ExcelWriter('oct_22_data.xlsx', engine='xlsxwriter')
oct_2022.to_excel(writer_oct_22, sheet_name = '2022 Oct', index=False)
writer_oct_22.sheets['2022 Oct'].set_column('A:A', 20)
writer_oct_22.close()



for day in days[:30]:
    nov_22_data = caiso.get_historical_demand(f'Nov {day}, 2022') 
    
    nov_2022 = pd.concat([nov_2022, nov_22_data], ignore_index=True)
    
nov_2022['Time'] = nov_2022['Time'].dt.tz_localize(None)

writer_nov_22 = pd.ExcelWriter('nov_22_data.xlsx', engine='xlsxwriter')
nov_2022.to_excel(writer_nov_22, sheet_name = '2022 Nov', index=False)
writer_nov_22.sheets['2022 Nov'].set_column('A:A', 20)
writer_nov_22.close()



for day in days:
    dec_22_data = caiso.get_historical_demand(f'Dec {day}, 2022') 
    
    dec_2022 = pd.concat([dec_2022, dec_22_data], ignore_index=True)
    
dec_2022['Time'] = dec_2022['Time'].dt.tz_localize(None)

writer_dec_22 = pd.ExcelWriter('dec_22_data.xlsx', engine='xlsxwriter')
dec_2022.to_excel(writer_dec_22, sheet_name = '2022 Dec', index=False)
writer_dec_22.sheets['2022 Dec'].set_column('A:A', 20)
writer_dec_22.close()



for day in days:
    jan_23_data = caiso.get_historical_demand(f'Jan {day}, 2023') 
    
    jan_2023 = pd.concat([jan_2023, jan_23_data], ignore_index=True)
    
jan_2023['Time'] = jan_2023['Time'].dt.tz_localize(None)

writer_jan_23 = pd.ExcelWriter('jan_23_data.xlsx', engine='xlsxwriter')
jan_2023.to_excel(writer_jan_23, sheet_name = '2023 Jan', index=False)
writer_jan_23.sheets['2023 Jan'].set_column('A:A', 20)
writer_jan_23.close()



for day in days[:28]:
    feb_23_data = caiso.get_historical_demand(f'Feb {day}, 2023') 
    
    feb_2023 = pd.concat([feb_2023, feb_23_data], ignore_index=True)
    
feb_2023['Time'] = feb_2023['Time'].dt.tz_localize(None)

writer_feb_23 = pd.ExcelWriter('feb_23_data.xlsx', engine='xlsxwriter')
feb_2023.to_excel(writer_feb_23, sheet_name = '2023 Feb', index=False)
writer_feb_23.sheets['2023 Feb'].set_column('A:A', 20)
writer_feb_23.close()



for day in days:
    mar_23_data = caiso.get_historical_demand(f'Mar {day}, 2023')
    
    mar_2023 = pd.concat([mar_2023, mar_23_data], ignore_index=True)
    
mar_2023['Time'] = mar_2023['Time'].dt.tz_localize(None)

writer_mar_23 = pd.ExcelWriter('mar_23_data.xlsx', engine='xlsxwriter')
mar_2023.to_excel(writer_mar_23, sheet_name = '2023 Mar', index=False)
writer_mar_23.sheets['2023 Mar'].set_column('A:A', 20)
writer_mar_23.close()



for day in days[:30]:
    apr_23_data = caiso.get_historical_demand(f'Apr {day}, 2023')
    
    apr_2023 = pd.concat([apr_2023, apr_23_data], ignore_index=True)
    
apr_2023['Time'] = apr_2023['Time'].dt.tz_localize(None)

writer_apr_23 = pd.ExcelWriter('apr_23_data.xlsx', engine='xlsxwriter')
apr_2023.to_excel(writer_apr_23, sheet_name = '2023 Apr', index=False)
writer_apr_23.sheets['2023 Apr'].set_column('A:A', 20)
writer_apr_23.close()



for day in days:
    may_23_data = caiso.get_historical_demand(f'May {day}, 2023')
    
    may_2023 = pd.concat([may_2023, may_23_data], ignore_index=True)
    
may_2023['Time'] = may_2023['Time'].dt.tz_localize(None)

writer_may_23 = pd.ExcelWriter('may_23_data.xlsx', engine='xlsxwriter')
may_2023.to_excel(writer_may_23, sheet_name = '2023 May', index=False)
writer_may_23.sheets['2023 May'].set_column('A:A', 20)
writer_may_23.close()



for day in days[:30]:
    jun_23_data = caiso.get_historical_demand(f'Jun {day}, 2023')
    
    jun_2023 = pd.concat([jun_2023, jun_23_data], ignore_index=True)
    
jun_2023['Time'] = jun_2023['Time'].dt.tz_localize(None)

writer_jun_23 = pd.ExcelWriter('jun_23_data.xlsx', engine='xlsxwriter')
jun_2023.to_excel(writer_jun_23, sheet_name = '2023 Jun', index=False)
writer_jun_23.sheets['2023 Jun'].set_column('A:A', 20)
writer_jun_23.close()



for day in days:
    jul_23_data = caiso.get_historical_demand(f'Jul {day}, 2023')
    
    jul_2023 = pd.concat([jul_2023, jul_23_data], ignore_index=True)
    
jul_2023['Time'] = jul_2023['Time'].dt.tz_localize(None)

writer_jul_23 = pd.ExcelWriter('jul_23_data.xlsx', engine='xlsxwriter')
jul_2023.to_excel(writer_jul_23, sheet_name = '2023 Jul', index=False)
writer_jul_23.sheets['2023 Jul'].set_column('A:A', 20)
writer_jul_23.close()



for day in days:
    aug_23_data = caiso.get_historical_demand(f'Aug {day}, 2023')
    
    aug_2023 = pd.concat([aug_2023, aug_23_data], ignore_index=True)
    
aug_2023['Time'] = aug_2023['Time'].dt.tz_localize(None)

writer_aug_23 = pd.ExcelWriter('aug_23_data.xlsx', engine='xlsxwriter')
aug_2023.to_excel(writer_aug_23, sheet_name = '2023 Aug', index=False)
writer_aug_23.sheets['2023 Aug'].set_column('A:A', 20)
writer_aug_23.close()



for day in days[:30]:
    sep_23_data = caiso.get_historical_demand(f'Sep {day}, 2023')
    
    sep_2023 = pd.concat([sep_2023, sep_23_data], ignore_index=True)
    
sep_2023['Time'] = sep_2023['Time'].dt.tz_localize(None)

writer_sep_23 = pd.ExcelWriter('sep_23_data.xlsx', engine='xlsxwriter')
sep_2023.to_excel(writer_sep_23, sheet_name = '2023 Sep', index=False)
writer_sep_23.sheets['2023 Sep'].set_column('A:A', 20)
writer_sep_23.close()



for day in days:
    oct_23_data = caiso.get_historical_demand(f'Oct {day}, 2023')
    
    oct_2023 = pd.concat([oct_2023, oct_23_data], ignore_index=True)
    
oct_2023['Time'] = oct_2023['Time'].dt.tz_localize(None)

writer_oct_23 = pd.ExcelWriter('oct_23_data.xlsx', engine='xlsxwriter')
oct_2023.to_excel(writer_oct_23, sheet_name = '2023 Oct', index=False)
writer_oct_23.sheets['2023 Oct'].set_column('A:A', 20)
writer_oct_23.close()


# workbook = 'CAISO_Historical_Data.xlsx'
# worksheets = workbook.worksheets()
# worksheets.sort(key=lambda sheet: sheet.get_name())
# for i, sheet in enumerate:
    # workbook.set_sheet_position(sheet.get_name(), i)
# workbook.close()





    

    




