import geopandas
from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from mpl_toolkits.basemap import Basemap
import fiona
from pykrige import OrdinaryKriging
from shapely.geometry import Polygon, Point
import sklearn.metrics as metrics


def result(y, y_hat):
    return (metrics.mean_absolute_error(y, y_hat),  # 0 表示完美预测，值越大表示预测误差越大。
            metrics.mean_squared_error(y, y_hat),
            sqrt(metrics.mean_squared_error(y, y_hat)),
            metrics.r2_score(y, y_hat))  # 1 表示完美预测，0 表示模型与简单平均值的效果相同，负值表示模型预测比直接使用平均值还要差。


workbook = gpd.read_file("random_data.csv", header=None)
# input_data = np.array(pd.read_csv("input.csv", header=None))
workbook = workbook.astype(float)
lon = workbook.field_3
lat = workbook.field_4
rainfall = workbook.field_73

# 中国的经纬度范围大约为：纬度3.86至53.55，经度73.66至135.05
# lon经度，lat纬度，40*1
un_lon = np.linspace(73.66, 135.05, 40)
un_lat = np.linspace(3.86, 53.55, 40)

OK = OrdinaryKriging(lon, lat, rainfall, variogram_model='gaussian', nlags=6)
zgrid, ss = OK.execute('grid', un_lon, un_lat)
# zgrid是插值结果，ss是插值的方差
# 定义为'grid'，则处理xpoints，ypoints为一个矩形网格的x、y坐标值列表，
# 定义为'point‘，则处理为坐标对的列表值。
# 定义为’masked‘，则处理xpoints，ypoints为矩形网格的x、y坐标，且采用掩膜（mask）评估特定的点值

xgrid, ygrid = np.meshgrid(un_lon, un_lat)

"""shp = fiona.open('shp/江苏省_行政边界.shp')
pol = next(iter(shp))  # 读取shp文件
polygon = Polygon(pol['geometry']['coordinates'][0][0])  # 这里创建了一个多边形对象，利用刚才的shp文件

# np.nan
for i in range(xgrid.shape[0]):
    for j in range(xgrid.shape[1]):
        plon = xgrid[i][j]
        plat = ygrid[i][j]
        if not polygon.contains(Point(plon, plat)):
            zgrid[i][j] = np.nan
"""
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130, facecolor='white')
base_map = Basemap(
    llcrnrlon=73.66,  # 地图左边的经度
    urcrnrlon=135.05,  # 地图右边的经度
    llcrnrlat=3.86,  # 地图最下面的纬度
    urcrnrlat=53.55,  # 地图最上面的纬度
    lon_0=105,  # 中心经度
    lat_0=29,  # 中心纬度
    ax=ax  # 画布
)
# 画经纬线，labels是经纬线的标签，fontsize是字体大小
# labels用于控制是否在平行线相交处标注标签列表（默认值为 [0,0,0,0]）。例如，labels=[1,0,0,1] 将导致平行线在与左侧和和底部相交的地方标注平行线，而不标注右侧和顶部。
# base_map.drawparallels(np.arange(73, 135, 8), labels=[1, 0, 0, 0], fontsize=12, ax=ax)
# base_map.drawmeridians(np.arange(4, 53, 8), labels=[0, 0, 0, 1], fontsize=12, ax=ax)
base_map.readshapefile(r"E:\code\gis\geo-master\shp\CNH-shp\gadm41_CHN_1", 'Js', True,
                       default_encoding='ISO-8859-1')
cp = base_map.pcolormesh(xgrid, ygrid, zgrid, cmap='Spectral_r', shading='auto')  # 伪彩色图
colorbar = base_map.colorbar(cp, label='IDW')  # 色带条
base_map.contour(xgrid, ygrid, zgrid, alpha=0.5, colors='white')  # 等高线图
#plt.axis('off')
# plt.figure()
# plt.scatter(xgrid, ygrid, label='Observations')  # 散点图只是显示了插值点
plt.show()

prediction_result = np.zeros(rainfall.shape)
for i in range(len(zgrid)):
    for j in range(len(lon)):  # i,j都是从0开始
        if zgrid[i][1] == lon[j] and zgrid[i][2] == lat[j]:
            prediction_result[j] = zgrid[i][0]

print(prediction_result, prediction_result.shape)
# prediction_result.to_csv("random_data.csv", index=False)
print("The accuracy:", result(rainfall, prediction_result))

print("zgrid is:", zgrid, "\nshape is:", zgrid.shape)
print("prediction result is:", prediction_result, "\nshape is:", prediction_result.shape)
print("The accuracy:", result(rainfall, prediction_result))
