import argparse
import rasterio as rs
from rasterio.windows import Window
import pickle
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon, Point
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import dist_btw_2node, generate_shp_some_nodes
from geopy import distance
from simulation_on_wind_graph import generate_df_edge,  network_original, get_node_by_type_cc, network_break_down, monte_carlo_simulation_till_converge
import pyproj
from shapely.ops import transform
from functools import partial
from shapely.ops import unary_union
import warnings
import matlab.engine

warnings.filterwarnings("ignore")

# def generate_circle_poly(lon, lat, radius):
#     local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(
#         lat, lon
#     )
#     wgs84_to_aeqd = partial(
#         pyproj.transform,
#         pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
#         pyproj.Proj(local_azimuthal_projection),
#     )
#     aeqd_to_wgs84 = partial(
#         pyproj.transform,
#         pyproj.Proj(local_azimuthal_projection),
#         pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
#     )

#     center = Point(float(lon), float(lat))
#     point_transformed = transform(wgs84_to_aeqd, center)
#     buffer = point_transformed.buffer(radius)
#     # Get the polygon with lat lon coordinates
#     circle_poly = transform(aeqd_to_wgs84, buffer)
#     return circle_poly


def update_fail_prob_for_gdf(gdf, folder, ii, threshold):
    V_THRESH = threshold*3600/1850
    V_HALF = 110
    wind_intensity = []
    tower_fs = []
    line_fs = []
    # src = rs.open(f'/scratch/users/zywei/cyclone_map/{tiff_name}.tif')
    # meta = src.meta
    moz = gpd.read_file(f'/scratch/users/zywei/cyclone_map/STORM/Moz_cyclone_track/{folder}')
    df = pd.read_csv(f'/scratch/users/zywei/cyclone_map/STORM/all_tracks/{folder}.csv')

    year = moz['Year'][ii]
    month = moz['Month'][ii]
    tcid = moz['TCID'][ii]
    tmp = df[(df['Year']==year)&(df['Month']==month)&(df['TCID']==tcid)]
    # print(len(tmp), events['count'][i])
    # circle_list = []
    speed_list = []

    cyclone_df_dict={'Lon':[],'Lat':[],'MaxSpeed':[],'Radius':[]}

    

    radius_last = -1
    for j in tmp.index:
        lon = tmp['Lon'][j]
        lat = tmp['Lat'][j]
        radius = tmp['Raius'][j]  # in m
        speed = tmp['MaxSpeed'][j]
        if radius_last>0:
            for i in range(1,6):
                lon_tmp = i/6.0*lon_last + (1-i/6.0)*lon
                lat_tmp = i/6.0*lat_last + (1-i/6.0)*lat
                radius_tmp = i/6.0*radius_last + (1-i/6.0)*radius
                speed_tmp = i/6.0*speed_last + (1-i/6.0)*speed
                # circle_list.append(generate_circle_poly(lon_tmp, lat_tmp, radius_tmp*1000))
                speed_list.append(speed_tmp)
                # speed_dict[(lon_tmp, lat_tmp)]= speed_tmp
                cyclone_df_dict['Lon'].append(lon_tmp)
                cyclone_df_dict['Lat'].append(lat_tmp)
                cyclone_df_dict['MaxSpeed'].append(speed_tmp)
                cyclone_df_dict['Radius'].append(radius_tmp)

        # circle_list.append(generate_circle_poly(lon, lat, radius*1000))
        speed_list.append(speed)
        # speed_dict[(lon, lat)]= speed
        cyclone_df_dict['Lon'].append(lon)
        cyclone_df_dict['Lat'].append(lat)
        cyclone_df_dict['MaxSpeed'].append(speed)
        cyclone_df_dict['Radius'].append(radius)
        lon_last = lon
        lat_last = lat
        radius_last = radius
        speed_last = speed
    #     mergedPolys = unary_union(circle_list)
    # print(mergedPolys)

    # cyclone_poly = mergedPolys #moz['geometry'][ii]
    cyclone_df = pd.DataFrame(cyclone_df_dict)
    # df0 =pd.DataFrame()
    # df0['speed'] = speed_list
    # gdf0 = gpd.GeoDataFrame(
    #     df0, geometry=circle_list, crs="EPSG:4326"
    # )
    # gdf0.to_file(f'/scratch/users/zywei/wind_result/wind_track/circle_track_{track_id}')
    # cyclone_df = df[(df['Year']==moz['Year'][ii])&(df['Month']==moz['Month'][ii])&(df['TCID']==moz['TCID'][ii])]
    
    moz_0 = gpd.read_file('/scratch/users/zywei/World_Countries')
    moz_geometry =  moz_0[moz_0['COUNTRY']=='Mozambique']['geometry'][151]

    eng = matlab.engine.start_matlab()
    eng.cd(r'ER11E04', nargout=0)
    wind_df = pd.DataFrame(index=gdf.index)
    for j in tqdm(cyclone_df.index):
        pt2 = (cyclone_df['Lat'][j], cyclone_df['Lon'][j])
        p = Point([ cyclone_df['Lon'][j], cyclone_df['Lat'][j]])
        if not moz_geometry.contains(p):
            continue
        Vmax = cyclone_df['MaxSpeed'][j]*1.0
        rmax = cyclone_df['Radius'][j] * 1000.0
        # [rr,VV,r0,rmerge,Vmerge,rrfracr0,MMfracM0,rmaxr0,MmM0,rmerger0,MmergeM0] = 
        # ER11E04_nondim_rmaxinput(70,70000,5e-5,0,1.5e-3,2/1000,0,1,0,0.15)
        #    ER11E04_nondim_rmaxinput(Vmax,rmax,fcor,Cdvary,C_d,w_cool,CkCdvary,CkCd,eye_adj,alpha_eye)

        

        t = eng.ER11E04_nondim_rmaxinput(Vmax,rmax,5e-5,0,1.5e-3,2/1000,0,1.0,0,0.15,nargout=11)
        rr = np.array(t[0]).reshape(-1)
        VV = np.array(t[1]).reshape(-1)
        # print(rr)
        # print(VV)


        winds = []
        for i in tqdm(gdf.index):
             

            pt = gdf['geometry'][i]
            if not pt:
                winds.append(None)
                continue
            pt1 = (pt.y, pt.x)
            d = distance.distance(pt1, pt2).km * 1000
            if d > rr[-1]:
                w = 0
            else:
                w = np.interp(d, rr, VV)
            winds.append(w)
        wind_df[j] = winds

        

        # wind_intensity.append(w0)
        # v = w0*3600/1850
        # v_n = max(0,v-V_THRESH)/(V_HALF-V_THRESH)
        # tower_f = v_n**3/(1+v_n**3)
        # tower_fs.append(tower_f)
        # v_tmp = min(w0, threshold+30)
        # v_tmp = max(threshold,v_tmp)
        # line_f = np.exp(0.6931*(v_tmp-threshold)/30)-1
        # #float(v_tmp-18)/30.0
        # line_fs.append(line_f)
    wind_intensity = wind_df.max(axis=1)

    v = wind_intensity*3600/1850
    tmp = (v-V_THRESH)/(V_HALF-V_THRESH)
    v_n = tmp.clip(0,np.inf)
    tower_fs = v_n**3/(1+v_n**3)

    # v_tmp = wind_intensity.clip(threshold,threshold+30)
    # line_fs = np.exp(0.6931*(v_tmp-threshold)/30)-1

    gdf[f'{folder[-1]}_{track_id}_w']=wind_intensity
    gdf[f'{folder[-1]}_{track_id}_b']=tower_fs
    # gdf['line_frac_prop_lost']=line_fs

def add_edge_attribute_for_simplified_network(gdf, MG_hospital, simplified_hospital_MG):
    node2tower_fail={}
    node2line_fail={}
    for i in gdf.index:
        n = gdf['id'][i]
        node2tower_fail[n] = gdf['tower_frac_prop_lost'][i]
        node2line_fail[n] = gdf['line_frac_prop_lost'][i]
    fail_prob_edges = {}

    for e in tqdm(simplified_hospital_MG.edges):
        path = nx.bidirectional_shortest_path(MG_hospital, e[0], e[1])
        tower_not_fail = 1
        for n in path[1:-1]:
            tower_not_fail *= (1-node2tower_fail[n])
        # fail_prob_edges[e] = 1 - tower_not_fail
        line_not_fail = 1
        for n1,n2 in zip(path[:-1],path[1:]):
            # print(n2)
            if n1 not in node2line_fail:
                line_not_fail *= 1 - node2line_fail[n2]
            elif n2 not in node2line_fail:
                line_not_fail *= 1 - node2line_fail[n1]
            else:
                line_not_fail *= 1 - max(node2line_fail[n1],node2line_fail[n2])
        fail_prob_edges[e] = {"tower_fail":1 - tower_not_fail, "line_fail": 1 - line_not_fail}
    nx.set_edge_attributes(simplified_hospital_MG, fail_prob_edges)

    e2length={}
    for e in tqdm(simplified_hospital_MG.edges):
        path = nx.bidirectional_shortest_path(MG_hospital, e[0], e[1])
        length = 0
        for n1,n2 in zip(path[:-1],path[1:]):
            length += dist_btw_2node(MG_hospital,n1,n2)
        e2length[e]={"length":length}
    nx.set_edge_attributes(simplified_hospital_MG, e2length)

if __name__ == "__main__":
    np.random.seed(seed=1234)

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--folder',  type=str)

    # parser.add_argument('--track_id',  type=int)
    parser.add_argument('--threshold',  type=int, default=25)
    args = parser.parse_args()
    # track_id = args.track_id
    folder = args.folder
    threshold = args.threshold

    fd2num = {'STORM_DATA_IBTRACS_SI_1000_YEARS_0': 290,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_1': 274, #274
            'STORM_DATA_IBTRACS_SI_1000_YEARS_2': 263,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_3': 276,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_4': 273,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_5': 279,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_6': 251,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_7': 264,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_8': 242,
            'STORM_DATA_IBTRACS_SI_1000_YEARS_9': 291
            }

    hospital = gpd.read_file('SARA_processed')
    public_net = hospital#[hospital['ele_main_s']==1]
    nodes_list = []
    for  i in tqdm(public_net.index):
        nodes_list.append(f'node/46774825{str(i).zfill(4)}')
    public_net['node_id'] = nodes_list

    for track_id in range(fd2num[folder]):
        print(track_id)
        update_fail_prob_for_gdf(public_net, folder, track_id, threshold)
    print(public_net)
    public_net.to_file(f"/scratch/users/zywei/wind_result/synthetic_new_hospital/track_{folder}")

  