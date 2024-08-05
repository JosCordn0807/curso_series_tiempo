# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:52:08 2024

@author: josea
"""


## Análisis descriptivo

import numpy as np
import os
import pandas as pd
from functools import reduce
from itertools import product
from statsmodels.tsa.stattools import adfuller, kpss
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #Gráficos Autocorrelación
from statsmodels.tsa.arima_process import ArmaProcess #Simulación del proceso ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX #Ajuste del modelo
#Pruebas estadísticas
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors, het_arch
#Descomposición
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import ccf
from pmdarima.arima import auto_arima

def plotseasonal(descomp):
    fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12,5))
    descomp.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observado')
    descomp.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Tendencia')
    descomp.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Estacionalidad')
    descomp.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residuos')
    
#Lectura

carpeta_archivos = r'C:\Users\josea\Documents\GitHub\curso_series_tiempo\Datos'
balance = pd.read_excel(os.path.join(carpeta_archivos, "balance_datos_completos.xlsx"))

#Index
balance.index = pd.DatetimeIndex(balance.fecha, freq = "MS")
balance.drop(columns = ['fecha'], inplace = True)

# Estacionariedad
var_dep = balance['oblig_publico']
adfuller(var_dep)[1]

# Mínimo d para ser estacionario

def min_d(serie, max_d = 12):
    pvalue = 1
    d = 0
    while (pvalue > 0.05) & (d <= max_d):
        eps_diff = np.diff(serie, n=d)
        pvalue = adfuller(eps_diff)[1]
        if(pvalue <= 0.05):
            print(f"Lag: {d} p-valor: {pvalue}") 
        d += 1
        
for col in balance.columns:
    print(col)
    min_d(balance[col])

var_dep_ld = np.log(var_dep).diff().dropna()

cols_exogen = ['tasa_desempleo', 'imae_var_ia', 'ipc_var_ia', 'tc_var_ia', 'tc_monex_var_ia', 'tbp']
var_exog_diff = balance[cols_exogen].apply(lambda x: x.diff(), axis = 0).dropna()

#Gráficos

var_dep.plot()
plt.title("Obligaciones con el público");

var_dep_diff = var_dep.diff().dropna()
var_dep_diff.plot()
plt.title("Obligaciones con el público (1° diferencia)");

var_dep_ld.plot()
plt.title("Obligaciones con el público (Var. porcentual)");

# ACF y PACF

plot_acf(var_dep_diff)
plot_pacf(var_dep_diff)

plot_acf(var_dep_ld)
plot_pacf(var_dep_ld)

# Estacionalidad

descomposicion = seasonal_decompose(var_dep, period = 12)
plotseasonal(descomposicion)

# CCF

#Prueba
x = balance['imae_var_ia']
y = balance['oblig_publico']
backwards = ccf(y, x, adjusted=False,nlags=20)[::-1]
forwards = ccf(x, y, adjusted=False,nlags=20)
ccf_output = np.r_[backwards[:-1], forwards]
  
def plot_ccf(x, y):
    backwards = ccf(y, x, adjusted=False,nlags=20)[::-1]
    forwards = ccf(x, y, adjusted=False,nlags=20)
    ccf_output = np.r_[backwards[:-1], forwards]
    plt.stem(range(-len(ccf_output)//2+1, len(ccf_output)//2+1), ccf_output)
    plt.title('CCF '+ x.name+' y '+y.name)
    plt.xlabel('Rezago')
    plt.ylabel('ACF')
    plt.axhline(-1.96/np.sqrt(len(x)), color='k', ls='--') 
    plt.axhline(1.96/np.sqrt(len(x)), color='k', ls='--')
    plt.show()
    
for col in cols_exogen:
    plot_ccf(var_exog_diff[col], var_dep_diff)  
    
def mejor_lag(x, y, max_lag = 12):
    backwards = ccf(y, x, adjusted=False,nlags=max_lag+1)[::-1]
    max_ccf = np.max(np.abs(backwards))
    if (max_ccf >= 1.96/np.sqrt(len(x))):
        posicion = np.argmax(np.abs(backwards))
        return max_lag - posicion
    else:
        return 0
   
var_exog_diff.apply(lambda x: mejor_lag(x, var_dep_diff))
    
var_exog_diff_bl = var_exog_diff.apply(lambda x: x.shift(mejor_lag(x, var_dep_diff))).dropna()
      
  
# Correlación

corr_exogen = var_exog_diff.dropna().corr()*100
sns.heatmap(corr_exogen, annot=True, fmt='.1f', vmin = -100, vmax = 100,cmap="coolwarm")
plt.show()

corr_exogen = var_exog_diff_bl.dropna().corr()*100
sns.heatmap(corr_exogen, annot=True, fmt='.1f', vmin = -100, vmax = 100,cmap="coolwarm")
plt.show()

# Modelos

var_dep_y = var_dep[var_exog_diff_bl.index.to_series()]
y_train, y_test = var_dep_y[:-6], var_dep_y.tail(6)
X_train, X_test = var_exog_diff_bl[:-6], var_exog_diff_bl.tail(6)

modelo1 = auto_arima(y = y_train, X = X_train,m = 12, d = 1)
modelo1.summary()
modelo1.plot_diagnostics(figsize=(10,8));

# Multicolinealidad

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(X_train)
pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 
              index=X.columns)

var_exog_diff_bl.drop(columns = ['tc_var_ia'], inplace=True)
X_train, X_test = var_exog_diff_bl[:-6], var_exog_diff_bl.tail(6)

modelo1 = auto_arima(y = y_train, X = X_train,m = 12, d = 1)
modelo1.summary()
modelo1.plot_diagnostics(figsize=(10,8));

def rolling_forecast(endog, exog, train_len, horizon, order, 
                     seasonal_order = (0,0,0,0), disp = False, with_intercept = True):
    data = np.array(endog)
    total_len = train_len + horizon
    pred_SARIMA = []
    trend = "n"
    if with_intercept:
        trend = "c"
    for i in range(train_len, total_len, 1):
        if disp:
            print(len(data[:i]))
        model = SARIMAX(data[:i],
                        order=order,
                        seasonal_order=seasonal_order, 
                        simple_differencing=False,
                       trend = trend)
        res = model.fit(disp=False)
        predictions = res.forecast(1)
        pred_SARIMA.extend(predictions)
        # EN CONSTRUCCIÓN
        if(i >= len(endog)):
            if disp:
                print("Ingreso nuevo dato")
            data = np.append(data,np.array(predictions))
    return pred_SARIMA

import itertools

def all_combinations(any_list):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i + 1)
        for i in range(len(any_list)))
lista_variables = [list(l) for l in all_combinations(var_exog_diff_bl.columns)]

signos = pd.Series([-1,1,-1,-1,-1],
                   index = ['tasa_desempleo', 'imae_var_ia', 'ipc_var_ia', 'tc_monex_var_ia', 'tbp'],
                   name = "signo_priori")

lista_modelos = []
for var in lista_variables:
    mod = auto_arima(y = y_train, X = X_train[var], m = 12,
                     max_p=11, max_q=11, max_order=24, max_P=4, max_Q = 4)
    coef = pd.Series(np.sign(mod.params()),
                     name = "signo_modelo")
    coef = coef[coef.index.isin(var_exog_diff_bl.columns)]
    coef = pd.merge(coef,signos, how="left", left_index = True, right_index = True)
    if (all(coef['signo_modelo'] * coef['signo_priori'] >= 0)):
        lista_modelos.append(mod)
        
mejor_modelo = np.argmin([mod.aic() for mod in lista_modelos])    
mejor_modelo = lista_modelos[mejor_modelo]

mejor_modelo.summary()
mejor_modelo.plot_diagnostics()

import pmdarima.arima as pm
pm.ADFTest().should_diff(var_dep_y.diff().dropna())
# Canova-Hansen 
pm.CHTest(m = 12).estimate_seasonal_differencing_term(var_dep_y.diff().dropna())
pm.OCSBTest(m=12).estimate_seasonal_differencing_term(var_dep_y.diff().dropna())

#Para tests
mod_test = SARIMAX(endog = y_train, exog = X_train['tbp'], order = (0,1,0), d = 1)