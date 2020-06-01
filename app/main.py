#!flask/bin/python
# Python
import os
import requests
import json
import MySQLdb
from datetime import date, datetime, timedelta
# FLASK
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
# Data Science
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle, geodesic
# Para geoposicion
from shapely import affinity
from shapely.geometry import MultiPoint, Point, LinearRing, Polygon


# Declaramos el APP Flask
app = Flask(__name__)

CORS(app)

# Servidores de Mysql

sv1 = MySQLdb.connect(host="ip", port=3306,
                      user="USUARIO", passwd="password", db="database_name")



def calculo():
    """ Metodo para el calculo de los puntos de paradas frecuentes o puntos de calor """

    # Consultamos la data para el calculos de las paradas frecuentes
    sv1.ping(True)
    cur = sv1.cursor()
    cur.execute(
        """ SELECT id, plate, timestamp AS fecha, latitude as lat, longitude as lon, timeOff FROM RuteoDynamic2 WHERE timestamp>1579496400 """)
    data = cur.fetchall()

    itms = []
    for row in data:
        dtPE = row[2]-18000
        hora = roundDatetime(datetime.fromtimestamp(
            dtPE), timedelta(minutes=30)).strftime("%H.%M")
        itms.append({
            'id': row[0],
            'plate': row[1],
            'fecha': datetime.fromtimestamp(dtPE).strftime("%d/%m/%Y %H:%M:%S"),
            'hora':  float(hora),
            'lat': row[3],
            'lon': row[4]
        })
    cur.close()
    # Obtenemos un DataFrame de la consulta con Pandas
    df = pd.DataFrame(itms)

    # Asignamos valores para el calculo
    coords = df[['lat', 'lon']].values
    coords2 = df[['lat', 'lon', 'hora', 'plate']].values

    # Se realiza los clustering de los puntos sercanos en un radio no mayor a 100m (Paradas frecuentes)
    kms_per_radian = 6371.0088
    epsilon = 0.07 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n]
                          for n in range(num_clusters)])
    clusters2 = pd.Series([coords2[cluster_labels == n]
                           for n in range(num_clusters)])
    poligonos = pd.DataFrame({'poligon': clusters2})

    # Buscamos el punto mas central entre los puntos mas frecuentes de paradas (centroides)
    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x,
                    MultiPoint(cluster).centroid.y)
        centermost_point = min(
            cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    # Asignamos
    centermost_points = clusters.map(get_centermost_point)

    # Los definimos en un dicionario de latitudes y longitudes
    lats, lons = zip(*centermost_points)

    # Lo adicionamos en un dataframe para manipularlo
    rep_points = pd.DataFrame({'lat': lats, 'lon': lons})

    # Anexamos sus atributos correspondientes
    rs = rep_points.apply(lambda row: df[(df['lat'] == row['lat']) & (
        df['lon'] == row['lon'])].iloc[0], axis=1)

    horas = []
    horas_s = []
    plates = []
    for n in range(num_clusters):
        x = pd.DataFrame(poligonos['poligon'][n])[[2]].values
        moda, count = stats.mode(poligonos['poligon'][n])
        horas.append(moda[0][2])
        horas_s.append(clusterTime(x))
        plates.append(moda[0][3])
        # print(moda)
    indice_moda = pd.DataFrame({
        'moda': horas,
        'moda_s': horas_s,
        'moda_plate': plates})

    # Obtenemos los puntos de calor o paradas frecuentes y lo exportamos a un json para mostrar
    datafinal = pd.concat([rs, indice_moda, poligonos],
                          axis=1).sort_values(by='moda')
    # Eliminamos la data que no cumpla con la condicion de minimo 4 puntos en su poligono
    datafinal.drop(
        datafinal[datafinal.poligon.str.len() <= 3].index, inplace=True)
    # datafinal.drop(['poligon'], axis=1)
    datafinal.to_csv('puntos.csv', header=True, index=False)
    return datafinal


@app.route('/')
def index():
    """ index """
    w = {
        'msg': 'Api Ruteo Dynamic!'
    }

    return jsonify(w)


@app.route('/loader')
def loader():
    """ Loader """
    csv = calculo()
    w = {
        'msg': 'Api Ruteo Dynamic, Calculo Hecho!'
    }

    return jsonify(w)


@app.route('/puntos')
def puntos():
    """ Muestra todos los puntos de paradas globales dentro de la mega geozona """
    Export = []

    # Total de puntos generados por el algoritmo de cluster
    puntos = pd.read_csv('puntos.csv')

    # Iteración por filas del DataFrame:
    for i, row in puntos.iterrows():
        Export.append({
            "type": "Feature",
            "properties": {
                "name": "Parada Frecuente",
                "moda": str(row[6]).replace(".", ":")+"0",
                "moda_s": str(row[7]).replace(".", ":")+"0",
                "moda_plate": row[8]
                # "poligon": np.array(row[8]).tolist()
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row[5], row[4]]
            }
        })

    return jsonify(Export)


@app.route('/puntos/cda/<cda>')
def cda_n(cda):
    """ Muestra todos los puntos de paradas globales dentro de la mega geozona """
    Export = []
    today = date.today().strftime("%d/%m/%Y")

    # Total de puntos generados por el algoritmo de cluster
    puntos = pd.read_csv('puntos.csv')

    # Poligonos de la zona de reparto
    poligono = polyReparto('', today, cda)

    # Obtenemos las cordenadas del poligono de la union de las zonas de repartos por cda
    # poly = Polygon(PolyConvex(poligono))
    poly = affinity.scale(Polygon(PolyConvex(poligono)), xfact=1.5, yfact=1.5)

    # Poligono de la zona de reparto y le aumentamos 20% del su area para abarcar puntos sercanos a esa zona
    # poly = affinity.scale(Polygon(poligono), xfact=1.1, yfact=1.1)

    # Chequeamos que del universo de puntos frecuentes se  enceuntre dentro de nuestro poligono de reparto y filtramos
    dentro = puntos[puntos.apply(
        lambda row: poly.contains(Point(row.lat, row.lon)), axis=1)]

    # Imprimir cordenadas dentro de la zona de reparto
    dentro_a = dentro.apply(lambda row: puntos[(puntos['lat'] == row['lat']) & (
        puntos['lon'] == row['lon'])].iloc[0], axis=1).sort_values(by='moda')

    # Iteración por filas del DataFrame:
    for i, row in dentro_a.iterrows():
        Export.append({
            "type": "Feature",
            "properties": {
                "name": "Parada Frecuente",
                "moda": str(row[6]).replace(".", ":")+"0",
                "moda_s": str(row[7]).replace(".", ":")+"0",
                "moda_plate": row[8]
                # "poligon": np.array(row[8]).tolist()
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row[5], row[4]]
            }
        })

    return jsonify(Export)


@app.route('/puntos/<ruta>')
def reparto(ruta):
    """ Muestra todos los puntos de paradas dentro de la zona de reparto """
    Export = []
    # Poligono de la zona de reparto
    poligono = polyReparto(ruta)

    # Cargar puntos globales de parada en la mega geozona
    puntos = pd.read_csv('puntos.csv')

    # Poligono de la zona de reparto y le aumentamos 30% del su area para abarcar puntos sercanos a esa zona
    poly = affinity.scale(Polygon(poligono), xfact=1.2, yfact=1.2)

    # Chequeamos que del universo de puntos frecuentes se  enceuntre dentro de nuestro poligono de reparto y filtramos
    dentro = puntos[puntos.apply(
        lambda row: poly.contains(Point(row.lat, row.lon)), axis=1)]

    # Imprimir cordenadas dentro de la zona de reparto
    dentro_a = dentro.apply(lambda row: puntos[(puntos['lat'] == row['lat']) & (
        puntos['lon'] == row['lon'])].iloc[0], axis=1).sort_values(by='moda')

    # Asignamos a un json todos los puntos de paradas para mostrar
    # Export = dentro_a.to_json(orient='records')
    # Iteración por filas del DataFrame:
    for i, row in dentro_a.iterrows():
        Export.append({
            "type": "Feature",
            "properties": {
                "name": "Parada Frecuente",
                "moda": str(row[6]).replace(".", ":")+"0",
                "moda_s": str(row[7]).replace(".", ":")+"0",
                "moda_plate": row[8]
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row[5], row[4]]
            }
        })

    return jsonify(Export)


def roundDatetime(dt, delta):
    """ Funcion para redondiar las horas para normalizar
    el calculo de la moda de la hora frecuente.
     """
    return dt + (datetime.min - dt) % delta


def polyReparto(ruta, today="", cda=""):
    """ Metodo para la extracción del poligono de la
    zona de reparto en el servidor 4 (SIM). """

    sv4.ping(True)
    cur = sv4.cursor()
    # query = "SELECT * FROM I_Rutas_Zonas WHERE fecha_programada = %s" if today != "" else "SELECT * FROM I_Rutas_Zonas WHERE id_ruta = %s"
    if today != "":
        query = """
            SELECT zr.id, zr.id_ruta, GROUP_CONCAT(zr.vertices) vertice
            FROM I_Rutas_Zonas zr
            LEFT JOIN I_Rutas r ON (r.id=zr.id_ruta)
            LEFT JOIN I_Importacion i ON (i.id=r.id_importacion)
            LEFT JOIN M_Paneles p ON (p.cda_id = i.cda_id AND p.canal_id = i.canal_id AND p.panel_id = r.id_panel)
            WHERE
            i.fecha_programada= %s AND
            i.cda_id = %s AND
            p.tipo = "horizontal" AND
            p.canal_id ="tradicional"
         """
        cur.execute(query, (today, cda))
    else:
        query = "SELECT * FROM I_Rutas_Zonas WHERE id_ruta = %s"
        cur.execute(query, (ruta, ))

    data = cur.fetchall()
    itms = []
    itms.append([tuple(float(c) for c in itm.split("/"))
                 for itm in data[0][2].split(",")])
    cur.close()
    return itms[0]


def clusterTime(x):
    """ Clustering al tiempo de la parada """
    clustering = DBSCAN(eps=2, min_samples=3).fit(x)
    cluster_labels = clustering.labels_
    num_clusters = len(set(cluster_labels))

    clusters = pd.Series([x[cluster_labels == n]for n in range(num_clusters)])
    h = []
    for n in range(num_clusters):
        # for i in range(len(clusters[n])):
        moda = stats.mode(clusters[n])
        if len(moda[0]) != 0:
            h.append(moda[0][0][0])
    r = h[1] if len(h) > 1 else 0
    return r


def PolyConvex(poligono):
    """ Funcion para calcular el poligono con una serie 
    de cordenadas dada 
    """
    points = np.array(poligono)
    hull = ConvexHull(points)
    ps = set()
    for x, y in hull.simplices:
        ps.add(x)
        ps.add(y)
    ps = np.array(list(ps))
    p = pd.DataFrame(points)

    return p.iloc[ps].values


if __name__ == '__main__':
    if os.environ['ENVIRONMENT'] == 'production':
        app.run(port=80, host='0.0.0.0')
