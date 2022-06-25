# -*- coding: utf-8 -*-
"""
@author: Elsy
# 

Elsy Yuliana Silgado Rivera
ID: 502194
elsy.silgado@upb.edu.co
"""

# IMPORTAR LAS LIBRERIAS DE TRATAMIENTO DE DATOS 
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


#  IMPORTAR LAS LIBRERIAS DE GRAFICOS
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')


# IMPORTAR LIBRERIAS DE PROCESADO Y MODELADO
# ==============================================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score


# SE REALIZA LA CONFIGURACIÓN WARNINGS
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# DECLARAR LA FUNCION PARA ASÍ REUTILIZAR EL CODIGO CON EL FIN DE PINTAR UN DENDOGRAMA,
# ACEPTA COMO PARAMETRO EL MODELO Y DENTRO DE LA FUNCIÓN SE LLEVA A CABO UN BUCLE FOR
# PARA ESTABLECER CUAL ES LA DISPOSICIÓN DE DICHAS AGRUPACIONES

def plot_dendrogram(model, **kwargs):
    '''
    Esta función extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
# //LA FUNCIÓN PLOT//
    dendrogram(linkage_matrix, **kwargs)
    

# //SIMULACIÓN DE DATOS//
# ==============================================================================

# SE REALIZA UNA SIMULACIÓN DE DATOS MEDIANTE EL METODO MAKE_BLOBS
# PARA ESTE SUCESO DE 200 PUNTOS CON SUS 4 CENTROIDES

X, y = make_blobs(
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )
# ==============================================================================

# PRESENTA EL GRÁFICO DE LA UBICACÓN DE LOS PUNTOS DESDE LAS COORDENADAS (X y Y)
# SE ESTIPUTA LOS COLORES MEDIANTE EL RECORRIDO DEL BUCLE FOR

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();


# //ESCALADO DE DATOS//
# ==============================================================================
X_scaled = scale(X)


# SE IMPLEMENTAN 3 MODELOS DIFERENTES TODOS AGGLOMERATIVECLUSTERING 
# PERO CON LINKAGE DIFERENTE... COMPLETE, AVERAGE, WARD
# TODOS LOS MODELOS CON LA MEDIDA DE DISTANCIA EUCLIDEANA

# ==============================================================================
modelo_hclust_complete = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'complete',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_complete.fit(X=X_scaled)

modelo_hclust_average = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_average.fit(X=X_scaled)

modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

# SE PINTAN LOS DENDROGRAMAS DE LOS 3 MODELOS
# USANDO LA FUNCION PLOT_DENDROGRAM 
# ==============================================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
altura_corte = 6
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
ax.set_title("Distancia euclídea, Linkage ward")
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();

# PARA IDENTIFICAR EL NÚMERO ÓPTIMO DE CLUSTERS SE USA MÉTODO SILHOUETTE
#A PARTIR DE UN BUBLE FOR ENSAYANDO CUAL ES EL NUMERO SE EVALUAN LOS MODELOS
# DE CLUSTER MAS OPTIMO PARA LA MUESTRA, EN ESTE CASO SE INTENTA 
# DESDE 2 A 15 CLUSTER, EL RESULTADO SE ALMACENA EN LA LISTA VALORES_MEDIOS_SILHOUETTE
# SE ELIJE EL MAS OPTIMO DE LOS VALORES Y SE PINTA, EN ESTE CASO SOLO SE UTILIZA 
# EL MODELO CON LINKAGE WARD
# ==============================================================================
range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    n_clusters = n_clusters
             )

    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');

# DESPUES DE TODAS LAS EVALUACIONES PARA EL MODELO MAS OPTIMO
# SE ELIJE EL LINKAGE WARD CON 4 CLUSTER PARA LA MUESTRA

# ==============================================================================
modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            n_clusters = 4
                     )
modelo_hclust_ward.fit(X=X_scaled)


# PINTAR EL MODELO PARA DEFINIR COMO FUE EL RESULTADO DE CLUSTER 
#====================================================


labels = modelo_hclust_ward.fit_predict(X_scaled)


fig_pred, ax_pred = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax_pred.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][labels[i]],
        marker    = 'o',
        edgecolor = 'black', 
        label= "Grupo {}".format(labels[i])
    )
ax_pred.set_title('Datos simulados')
ax_pred.legend();