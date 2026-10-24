
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
from scipy.stats import norm
import math

warnings.filterwarnings("ignore")

def ms2mph(w):
    return 2.23694 * w
def mph2ms(w):
    return w/2.23694

def fm2gust3s(fm):
    return fm * 1.05 + 10.5

def prob_damage(w0):
    if w0 == 0:
       return 0
    gust3s = mph2ms(fm2gust3s(ms2mph(w0)))
    # gust3s = w0 * 1.64

    tmp = (math.log(gust3s)-math.log(80))/0.32
    return norm.cdf(tmp)

def ss_category(w):
    if w >= 70:
      return 5
    elif w >= 58:
      return 4
    elif w >= 50:
      return 3
    elif w >= 43:
      return 2
    elif w >= 33:
      return 1
    elif w >= 18:
      return 0
    else:
      return -1

def prob_irr(w0, d_min, r):
    C = ss_category(w0)
    R = d_min/r
    discr = R + (1.15 -0.126*C)/(2.48 - 0.139*C)
    if C == -1:
        return 0
    elif discr > 1:
        return 0
    else:
        tmp = (0.0965*C+1.97)*math.log(discr)
        return 1-math.exp(tmp)


def update_fail_prob_for_gdf(solar, folder, ii):

    moz = gpd.read_file(f'/scratch/users/zywei/cyclone_map/STORM/Moz_cyclone_track/{folder}')
    df = pd.read_csv(f'/scratch/users/zywei/cyclone_map/STORM/all_tracks/{folder}.csv')
    # print(moz)

    year = moz['Year'][ii]
    month = moz['Month'][ii]
    tcid = moz['TCID'][ii]
    tmp = df[(df['Year']==year)&(df['Month']==month)&(df['TCID']==tcid)]

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
                speed_list.append(speed_tmp)
                cyclone_df_dict['Lon'].append(lon_tmp)
                cyclone_df_dict['Lat'].append(lat_tmp)
                cyclone_df_dict['MaxSpeed'].append(speed_tmp)
                cyclone_df_dict['Radius'].append(radius_tmp)

        speed_list.append(speed)
        cyclone_df_dict['Lon'].append(lon)
        cyclone_df_dict['Lat'].append(lat)
        cyclone_df_dict['MaxSpeed'].append(speed)
        cyclone_df_dict['Radius'].append(radius)
        lon_last = lon
        lat_last = lat
        radius_last = radius
        speed_last = speed

    cyclone_df = pd.DataFrame(cyclone_df_dict)
    eng = matlab.engine.start_matlab()
    eng.cd(r'ER11E04', nargout=0)
    wind_df = pd.DataFrame(index=solar.index)
    irr_df = pd.DataFrame(index=solar.index)

    moz_0 = gpd.read_file('/scratch/users/zywei/World_Countries')
    moz_geometry =  moz_0[moz_0['COUNTRY']=='Mozambique']['geometry'][151]


    for j in cyclone_df.index:
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
        print(rr[-1])

        winds = []
        irrs = []
        idx = sum(rr<rmax)
        ROCI = np.interp(-2, -VV[idx:], rr[idx:])
        # print(ROCI, rr[-1],  VV[-1], 2, rr[-100], VV[-100])
        for i in solar.index:
            pt = solar['geometry'][i]
            pt1 = (pt.y, pt.x)
            d = distance.distance(pt1, pt2).km * 1000
            if d > rr[-1]:
                w = 0
            else:
                w = np.interp(d, rr, VV)
                w = max(0, w)
            winds.append(w)
            
            irrs.append(prob_irr(w, d, ROCI))

        wind_df[j] = winds
        irr_df[j] = irrs

    # Some synthetic tracks never place an interpolated center inside
    # Mozambique.  An empty dataframe max is NaN, but under this model such an
    # event has zero local wind and irradiance impact.
    if wind_df.shape[1] == 0:
        wind_intensity = pd.Series(0.0, index=solar.index)
        irr_decay = pd.Series(0.0, index=solar.index)
    else:
        wind_intensity = wind_df.max(axis=1)
        irr_decay = irr_df.max(axis=1)
    solar['wind_intensity'] = wind_intensity
    solar['irr_decay'] = irr_decay
    solar['damage']=solar['wind_intensity'].map(prob_damage)


if __name__ == "__main__":
    np.random.seed(seed=1234)

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--folder',  type=str)

    # parser.add_argument('--track_id',  type=int)
    parser.add_argument('--threshold',  type=int, default=24)
    args = parser.parse_args()
    # track_id = args.track_id
    folder = args.folder
    threshold = args.threshold
    

    sara = gpd.read_file('SARA_processed')
    # public_net = hospital[hospital['ele_main_s']==1]
    solar = sara[(sara['ele_main_s']==3)|(sara['ele_sec_so']==3)]
    solar = solar[~solar["geometry"].isna()]
    nodes_list = []
    for  i in tqdm(solar.index):
        nodes_list.append(f'node/46774825{str(i).zfill(4)}')
    solar['node_id'] = nodes_list
    solar_copy = solar.copy()
    solar_irr = solar.copy()
    solar_damage = solar.copy()

    fd2num = {'STORM_DATA_IBTRACS_SI_1000_YEARS_0': 290,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_1': 274, #274
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_2': 263,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_3': 276,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_4': 273,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_5': 279,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_6': 251,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_7': 264,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_8': 242,
                    'STORM_DATA_IBTRACS_SI_1000_YEARS_9': 291}
    
    for ii in tqdm(range(fd2num[folder])):
        solar_tmp = solar_copy.copy()
        update_fail_prob_for_gdf(solar_tmp, folder, ii)
        # solar_tmp.to_file(f'/scratch/users/zywei/wind_result/synthetic_solar/{folder}_var_track_{ii}_solar_func')
        # print(solar[solar["irr_decay"]>0])
        solar_irr[str(ii)]  = solar_tmp["irr_decay"]
        solar_damage[str(ii)]  = solar_tmp["damage"]

    # print(solar_irr)
    # for c in solar_irr.columns.tolist():
    #     print(c, type(c))
    solar_irr.to_file(f'/scratch/users/zywei/wind_result/synthetic_solar/{folder}_var_track_solar_irr_func_2')
    solar_damage.to_file(f'/scratch/users/zywei/wind_result/synthetic_solar/{folder}_var_track_solar_damage_func_2')



    





    # public_net = monte_carlo_simulation_till_converge(simplified_hospital_MG_wind, public_net, df_edge, s2mw, threshold, folder, track_id)
    # public_net.to_file(f'/scratch/users/zywei/wind_result/synthetic_new/{folder}_var_track_{track_id}_{threshold}_func')
 
