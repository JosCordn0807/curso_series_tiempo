# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:11:47 2024

@author: JoséAlbertoCordónCam
"""

import pandas as pd
import os 
import numpy as np
from functools import reduce

carpeta_archivos = r'C:\Users\josea\Documents\Insumos Series de Tiempo'

archivo_variables = os.path.join(carpeta_archivos, "Variables macroeconómicas.xlsx")

meses = {
    "enero": "1",
    "febrero": "2",
    "marzo": "3",
    "abril": "4",
    "mayo": "5",
    "junio": "6",
    "julio": "7",
    "agosto": "8",
    "septiembre": "9",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
    "ene": "1", "feb": "2",
    "mar": "3", "abr": "4",
    "may": "5", "jun": "6",
    "jul": "7", "ago": "8",
    "sep": "9", "oct": "10",
    "nov": "11", "dic": "12",
    "set": "9"
}

#TBP
tbp = pd.read_excel(archivo_variables, sheet_name="TBP", skiprows=4).rename(columns = {'Unnamed: 0': 'mes'})
tbp = pd.melt(tbp, id_vars='mes', var_name = "year", value_name = 'tbp')
tbp.mes = tbp.mes.str.lower().replace(meses)
tbp['fecha'] = [pd.to_datetime(f"{year}-{mes}") for mes, year in zip(tbp.mes, tbp.year)]
tbp.index = pd.PeriodIndex(tbp.fecha, freq="M")
tbp = tbp[['tbp']].dropna()

#IPC
ipc = (pd.read_excel(archivo_variables, sheet_name="IPC", skiprows=4)
       .rename(columns = {'Unnamed: 0': 'mes','Variación interanual (%)':'ipc_var_ia'}))
ipc.mes = ipc.mes.str.lower().apply(lambda x: reduce(
    lambda a, kv: a.replace(*kv), list(meses.items()), x))
ipc['fecha'] = pd.to_datetime(ipc.mes, format = "%m/%Y")
ipc.index = pd.PeriodIndex(ipc['fecha'], freq="M")
ipc = ipc[['ipc_var_ia']].dropna()

#Desempleo
desempleo = (pd.read_excel(archivo_variables, sheet_name="Desempleo", skiprows=4)
             .rename(columns = {'Unnamed: 0': 'valor'}))
desempleo.valor = desempleo.valor.str.strip()
desempleo = desempleo.loc[desempleo.valor == "Tasa de desempleo"]
desempleo = pd.melt(desempleo, id_vars='valor', var_name = "mes", value_name = 'tasa_desempleo')
desempleo.mes = desempleo.mes.str.lower().apply(lambda x: reduce(
    lambda a, kv: a.replace(*kv), list(meses.items()), x))
desempleo['fecha'] = pd.to_datetime(desempleo.mes, format = "%m/%Y")
desempleo.index = pd.PeriodIndex(desempleo['fecha'], freq="M")
desempleo = desempleo[['tasa_desempleo']].dropna()

#IMAE
imae = (pd.read_excel(archivo_variables, sheet_name="IMAE", skiprows=4)
        .rename(columns = {'Unnamed: 0': 'mes','Variación interanual':'imae_var_ia'}))
imae.mes = imae.mes.str.lower().apply(lambda x: reduce(
    lambda a, kv: a.replace(*kv), list(meses.items()), x))
imae['fecha'] = pd.to_datetime(imae.mes, format = "%m/%Y")
imae.index = pd.PeriodIndex(imae['fecha'], freq="M")
imae = imae[['imae_var_ia']].dropna()

#Tipo de cambio
tipo_cambio = pd.read_excel(archivo_variables, sheet_name="TC", skiprows=4).rename(
    columns={'Unnamed: 0': 'mes', 'TIPO DE CAMBIO VENTA': 'tipo_cambio'})
tipo_cambio.mes = tipo_cambio.mes.str.lower().apply(lambda x: reduce(
    lambda a, kv: a.replace(*kv), list(meses.items()), x))
tipo_cambio.dropna(inplace = True)
tipo_cambio['fecha'] = pd.to_datetime(tipo_cambio.mes, format="%d %m %Y")

#Versión 1A
rango_fechas = pd.date_range(min(tipo_cambio.fecha), max(tipo_cambio.fecha),freq = "M")
tipo_cambio = tipo_cambio.loc[tipo_cambio.fecha.isin(rango_fechas)]
tipo_cambio.index = pd.PeriodIndex(tipo_cambio['fecha'], freq="M")
tipo_cambio.sort_index(inplace = True)
tipo_cambio['tc_var_ia'] = (np.log(tipo_cambio.tipo_cambio)-np.log(tipo_cambio.tipo_cambio.shift(12))) * 100
tipo_cambio = tipo_cambio[['tc_var_ia']].dropna()


#%%
#Versión 1B
rango_fechas = tipo_cambio.groupby(by = [tipo_cambio.fecha.dt.month, tipo_cambio.fecha.dt.year]).fecha.max()

#Versión 2
tipo_cambio_prom = (tipo_cambio.groupby(by = [tipo_cambio.fecha.dt.month, tipo_cambio.fecha.dt.year])
                    .agg({'fecha':max, 'tipo_cambio':np.mean})
                    .reset_index(drop = True))
#%%
#Tipo de cambio MONEX
tc_monex = pd.read_excel(archivo_variables, sheet_name="TC_MONEX", skiprows=4).rename(
    columns={'Unnamed: 0': 'mes'})
tc_monex = pd.melt(tc_monex, id_vars='mes', var_name = "year", value_name = 'tc_monex')
tc_monex = tc_monex.loc[tc_monex['tc_monex'] > 0].dropna()
tc_monex.mes = tc_monex.mes.str.lower().apply(lambda x: reduce(
    lambda a, kv: a.replace(*kv), list(meses.items()), x))
tc_monex['fecha'] = [pd.to_datetime(f"{mes} {year}", format = "%d %m %Y") for mes, year in zip(tc_monex.mes, tc_monex.year)]
rango_fechas = tc_monex.groupby(by = [tc_monex.fecha.dt.month, tc_monex.fecha.dt.year]).fecha.max()
tc_monex = tc_monex.loc[tc_monex.fecha.isin(rango_fechas)]
tc_monex.index = pd.PeriodIndex(tc_monex['fecha'], freq="M")
tc_monex.sort_index(inplace = True)
tc_monex['tc_monex_var_ia'] = (np.log(tc_monex.tc_monex)-np.log(tc_monex.tc_monex.shift(12))) * 100
tc_monex = tc_monex[['tc_monex_var_ia']].dropna()


variables_macro = pd.concat([desempleo, imae, ipc, tipo_cambio, tc_monex, tbp], axis = 1).sort_index().dropna()

# Balances

balance0 = pd.read_excel(os.path.join(carpeta_archivos, "Balances.xlsx"), sheet_name="Antes Marzo 2020").dropna()
cuentas0 = ("251","274","290")
balance0 = balance0.loc[balance0.Periodo.str.startswith(cuentas0)]
balance0 = balance0.transpose()
balance0.columns = ['creditos','deterioro_creditos','oblig_publico']
balance0 = balance0.drop(["Periodo","Total"])
balance0.index = pd.PeriodIndex(pd.to_datetime(balance0.index), freq = "M")
balance0.index.name = "fecha"

cuentas1 = ("1515","1538","1555")
balance1 = pd.read_excel(os.path.join(carpeta_archivos, "Balances.xlsx"), sheet_name="Después Marzo 2020").dropna()
balance1 = balance1.loc[balance1.Periodo.str.startswith(cuentas1)]
balance1 = balance1.transpose()
balance1.columns = ['creditos','deterioro_creditos','oblig_publico']
balance1 = balance1.drop(["Periodo","Total"])
balance1.index = pd.PeriodIndex(pd.to_datetime(balance1.index), freq = "M")
balance1.index.name = "fecha"

balance = pd.concat([balance0, balance1])
del balance0, balance1
balance = pd.concat([balance, variables_macro], axis = 1).dropna()
balance.to_excel(os.path.join(carpeta_archivos,"balance_datos_completos.xlsx"))