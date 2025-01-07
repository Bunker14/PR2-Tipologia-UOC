import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import shapiro, levene, f_oneway, kruskal

import matplotlib.pyplot as plt


#####################################################################################
# 1	Integración y selección
# Importamos los datasets
#####################################################################################
print('\n\nDescripcion de los datasets:')

#Bitcoin
bitcoin = pd.read_csv('BTCUSDT.csv')
bitcoin['fecha']=pd.to_datetime(bitcoin['timestamp'])
bitcoin['año']=bitcoin['fecha'].dt.year
bitcoin_por_year = bitcoin.groupby('año').size()
print('\n Dataset Bitcoin:')
print(bitcoin_por_year.sort_values(ascending=False))

#Ethereum
ethereum = pd.read_csv('ETHUSDT.csv')
ethereum['fecha']=pd.to_datetime(ethereum['timestamp'])
ethereum['año']=ethereum['fecha'].dt.year
ethereum_por_year = ethereum.groupby('año').size()
print('\nDataset Ethereum:')
print(ethereum_por_year.sort_values(ascending=False))

#Litecoin
litecoin = pd.read_csv('LTCUSDT.csv')
litecoin['fecha']=pd.to_datetime(litecoin['timestamp'])
litecoin['año']=litecoin['fecha'].dt.year
litecoin_por_year = litecoin.groupby('año').size()
print('\nDataset Litecoin:')
print(litecoin_por_year.sort_values(ascending=False))


########################################################################################
# 2	Integración y selección:  Creamos el datasetFinal a partir de las archivos csv.
# Añadiremos nuevos campos como la Criptomoneda, el Año, MesDesc, DiaSemanaDesc, spread
# y un id de cada registro
########################################################################################

#Bitcoin
bitcoin['criptomoneda']='Bitcoin'
bitcoin_filtrado=bitcoin[(bitcoin['año']>=2023)&(bitcoin['año']<=2024)]

#Ethereum
ethereum['criptomoneda']='Ethereum'
ethereum_filtrado=ethereum[(ethereum['año']>=2023)&(ethereum['año']<=2024)]

#Litecoin
litecoin['criptomoneda']='Litecoin'
litecoin_filtrado=litecoin[(litecoin['año']>=2023)&(litecoin['año']<=2024)]


# Unimos los diferentes datasets en uno final y creamos la columna Id para tener todos los registros identificados

df=pd.concat([bitcoin_filtrado, ethereum_filtrado, litecoin_filtrado], ignore_index=True)

df['id'] = range(1, len(df) + 1)



print('\n\nEl dataset Final contiene: ' +str(len(df)))
# print(df.head())

# Creamos las columnas calculadas que necesitamos para el estudio

df['mesDesc']=df['fecha'].dt.month_name(locale='es_ES')
df['diaSemanaDesc']=df['fecha'].dt.day_name(locale='es_ES')
df['spread']=df['high']-df['low']

## tipoColumnasAntes=df.dtypes
## print(tipoColumnasAntes)

#Tratamos las columnas para que tengan el nombre y el formato adecuado

dfR=df.rename(columns={'number_of_trades':'numeroOperaciones'})

dfR['criptomoneda']=dfR['criptomoneda'].astype('category')
dfR['mesDesc']=dfR['mesDesc'].astype('category')
dfR['diaSemanaDesc']=dfR['diaSemanaDesc'].astype('category')

# Eliminamos todas las columnas que no sean necesarias para nuestro analisis

datasetAnalisis=dfR.drop(columns=['timestamp','high','low','quote_asset_volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore','close_time'])

tipoColumnasAnalisis=datasetAnalisis.dtypes
print("\n\nEl formato de los atributos del Dataset final:\n ")
print(tipoColumnasAnalisis)

resumen = datasetAnalisis
pd.set_option('display.max_columns', None)
# Resumen general
print("\n\nResumen de los datos seleccionados para análisis diario:\n")
print(resumen.describe(include='all'))

########################################################################################
# 3	Limpieza de los datos.  .
########################################################################################

### 3.1

# Revisar valores nulos y ceros en el dataset
nulos = datasetAnalisis.isnull().sum()
ceros = (datasetAnalisis == 0).sum()

# Mostrar los resultados en consola
print("\n\nValores Nulos en el Dataset:")
print(nulos)
print("\nValores Cero en el Dataset:")
print(ceros)


datasetAnalisis['volume'] = datasetAnalisis['volume'].replace(0, datasetAnalisis['volume'][datasetAnalisis['volume'] > 0].min())
datasetAnalisis['numeroOperaciones'] = datasetAnalisis['numeroOperaciones'].replace(0, datasetAnalisis['numeroOperaciones'][datasetAnalisis['numeroOperaciones'] > 0].min())

#Se crean dos datasets. dfa que será la base de datos sin perdidas y el dfap que se forzará la perdida de algunos valores

dfa=datasetAnalisis
dfap=datasetAnalisis

#### Se introducen 50 registros nulos dentro de las columnas

numNulos =50

# Atributo Open
indAleatorios = np.random.choice(dfap.index, numNulos, replace=False)
dfap.loc[indAleatorios, 'open'] = np.nan

# Atributo Close
indAleatorios2 = np.random.choice(dfap.index, numNulos, replace=False)
dfap.loc[indAleatorios2, 'close'] = np.nan



nulosPerdida = dfap.isnull().sum()

print("\n Valores Nulos en el Dataset Modificado:")
print(nulosPerdida)

#Reemplazamos los valores null

dfap['open'] = dfap['open'].fillna(df['open'].mean())

dfap['close'] = dfap['close'].fillna(df['close'].mean())


nulosPerdida2 = dfap.isnull().sum()

print("\n Valores Nulos en el Dataset Modificado y sustituidos:")
print(nulosPerdida2)


### 3.3 Identifica y gestiona los valores extremos.


data_extremos = dfap.copy()  #Cambiar el dataset según el estudio que se quiera realizar (dfa o dfap)


columns_to_check = ['open','close', 'volume', 'numeroOperaciones', 'spread']

extreme_analysis = []

for column in columns_to_check:
    Q1 = data_extremos[column].quantile(0.25)
    Q3 = data_extremos[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Contar valores extremos
    outliers_below = data_extremos[data_extremos[column] < lower_bound].shape[0]
    outliers_above = data_extremos[data_extremos[column] > upper_bound].shape[0]

    extreme_analysis.append({
        'Column': column,
        'Q1 (25%)': Q1,
        'Q3 (75%)': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers Below': outliers_below,
        'Outliers Above': outliers_above
    })

    # Convertir el análisis a un DataFrame para mostrar
    extreme_analysis_df = pd.DataFrame(extreme_analysis)
    data_extremos[column] = data_extremos[column].clip(lower=lower_bound, upper=upper_bound)

print("\n\n Valores extremos:\n")
print(extreme_analysis_df)


df_final = data_extremos

## df_final.to_csv('dfapFinal.csv',index=False,chunksize=100000) #Instruccion para guardar los datasets en csv


########################################################################################
# 4.	Análisis de los datos.
########################################################################################

### 4.1 a) Modelo Supervisado

X = df_final[['open', 'volume', 'numeroOperaciones', 'spread']]
y = df_final['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n\n Resultados Modelo Supervisado:\n")
print(f"MSE: {mse}, R2: {r2}")


### 4.1 b) Modelo NO Supervisado

X_clustering = df_final[['open', 'volume', 'numeroOperaciones', 'spread']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_final['cluster'] = clusters

print("\n\n Resultados Modelo NO Supervisado:\n")
print(f"Inercia: {kmeans.inertia_}")

### 4.1 c) Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales y los clusters
df_clusters = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_clusters['Cluster'] = clusters

# Visualizar los clusters
plt.figure(figsize=(8, 6))
for cluster in df_clusters['Cluster'].unique():
    cluster_data = df_clusters[df_clusters['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
plt.title('Clusters en 2D')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

### 4.2 prueba por contraste de hipótesis.

# Separar datos por criptomoneda

btc_close = df_final[df_final['criptomoneda'] == 'Bitcoin']['close']
eth_close = df_final[df_final['criptomoneda'] == 'Ethereum']['close']
ltc_close = df_final[df_final['criptomoneda'] == 'Litecoin']['close']

# Prueba de normalidad (Shapiro-Wilk)
shapiro_btc = shapiro(btc_close)
shapiro_eth = shapiro(eth_close)
shapiro_ltc = shapiro(ltc_close)

# Prueba de homocedasticidad (Levene)
levene_test = levene(btc_close, eth_close, ltc_close)

# Prueba de ANOVA o Kruskal-Wallis según los resultados
if all(p > 0.05 for p in [shapiro_btc.pvalue, shapiro_eth.pvalue, shapiro_ltc.pvalue]) and levene_test.pvalue > 0.05:
    # Si los datos son normales y homogéneos
    anova_test = f_oneway(btc_close, eth_close, ltc_close)
    test_type = 'ANOVA'
    test_result = anova_test
else:
    # Si los datos no son normales o no homogéneos
    kruskal_test = kruskal(btc_close, eth_close, ltc_close)
    test_type = 'Kruskal-Wallis'
    test_result = kruskal_test

# Resultados
{
    "Shapiro_Bitcoin": shapiro_btc,
    "Shapiro_Ethereum": shapiro_eth,
    "Shapiro_Litecoin": shapiro_ltc,
    "Levene": levene_test,
    "Test_Type": test_type,
    "Test_Result": test_result
}

# Imprimir los resultados de las pruebas de manera bonita

print("\n\nResultados del Contraste de Hipótesis:\n")

# Resultados del Shapiro-Wilk
print("1. Prueba de Normalidad (Shapiro-Wilk):")
print(f"   - Bitcoin: Estadístico = {shapiro_btc.statistic:.4f}, p-valor = {shapiro_btc.pvalue:.2e}")
print(f"   - Ethereum: Estadístico = {shapiro_eth.statistic:.4f}, p-valor = {shapiro_eth.pvalue:.2e}")
print(f"   - Litecoin: Estadístico = {shapiro_ltc.statistic:.4f}, p-valor = {shapiro_ltc.pvalue:.2e}")

# Resultado de la prueba de homocedasticidad
print("\n2. Prueba de Homocedasticidad (Levene):")
print(f"   - Estadístico = {levene_test.statistic:.4f}, p-valor = {levene_test.pvalue:.2e}")

# Resultado de la prueba de hipótesis final
print(f"\n3. Prueba de Hipótesis ({test_type}):")
print(f"   - Estadístico = {test_result.statistic:.4f}, p-valor = {test_result.pvalue:.2e}")
