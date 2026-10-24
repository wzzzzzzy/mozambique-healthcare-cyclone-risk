from geopy import distance
from tqdm import tqdm
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon, Point
import pandas as pd
import geopandas as gpd

def get_node_pos(MG, n):
    shp=shape(MG.nodes[n]['geometry'])
    pt = shp
    if type(shp)==Polygon:
        pt = shp.centroid
    elif type(shp)==Point:
        pt = shp
    return pt


def dist_btw_2Point(Pt1,Pt2):

  pt1 = (Pt1.y, Pt1.x)
  pt2 = (Pt2.y, Pt2.x)
  d = distance.distance(pt1, pt2).km
  return d

def dist_btw_2node(MG, n1,n2):
  Pt1 = get_node_pos(MG, n1)
  Pt2 = get_node_pos(MG, n2)

  pt1 = (Pt1.y, Pt1.x)
  pt2 = (Pt2.y, Pt2.x)
  d = distance.distance(pt1, pt2).km
  return d

def get_node_by_type(MG):
    type_set=set()
    node_by_type={}
    for n in MG.nodes:
        type_set.add(MG.nodes[n]['power'])
    for t in type_set:
        node_by_type[t]=[]
        for n in MG.nodes:
            if MG.nodes[n]['power']==t:
                node_by_type[t].append(n)
    return node_by_type

def get_simple(MG):
    simple = []
    node_by_type=get_node_by_type(MG)
    simple.extend(node_by_type['plant'])
    simple.extend(node_by_type['generator'])
    simple.extend(node_by_type['substation'])
    simple.extend(node_by_type['transformer'])
    simple.extend(node_by_type['transition'])
    simple.extend(node_by_type['portal'])
    return simple

def get_simple_connect(MG):
    simple = []
    node_by_type=get_node_by_type(MG)
    simple.extend(node_by_type['plant'])
    simple.extend(node_by_type['generator'])
    simple.extend(node_by_type['substation'])
    simple.extend(node_by_type['transformer'])
    simple.extend(node_by_type['transition'])
    simple.extend(node_by_type['portal'])
    simple.extend(node_by_type['gridfinder'])
    return simple

def generate_shp_some_nodes(MG, nodes_list):
    key_set = set()
    for n in nodes_list:
        for k in MG.nodes[n]['properties']:
            key_set.add(k)
    dict_sub = {}
    for k in key_set:
        dict_sub[k] = []
    geoms =[]
    ids = []
    for n in nodes_list:
        ids.append(n)
        for k in key_set:
            if k in MG.nodes[n]['properties']:
                dict_sub[k].append(MG.nodes[n]['properties'][k])
            else:
                dict_sub[k].append(-1)

        shp=shape(MG.nodes[n]['geometry'])
        pt = shp
        if type(shp)==Polygon:
            pt = shp.centroid
        elif type(shp)==Point:
            pt = shp
        geoms.append(pt)
    df =pd.DataFrame()
    for k in key_set:
        df[k] = dict_sub[k]
    df['id'] = ids
    gdf = gpd.GeoDataFrame(
        df, geometry=geoms, crs="EPSG:4326"
    )
    return gdf

def generate_indexing(MG):
    partition_pt={}
    id2loc={}
    for n in MG.nodes:
        shp=shape(MG.nodes[n]['geometry'])
        pt = shp

        if type(shp)==Polygon:
            pt = shp.centroid
        elif type(shp)==Point:
            pt = shp
        id2loc[n]=pt
        if (int(pt.x),int(pt.y)) not in partition_pt:
            partition_pt[(int(pt.x),int(pt.y))]=[]
        partition_pt[(int(pt.x),int(pt.y))].append(n)
    return partition_pt, id2loc

def generate_indexing_specify(MG, node_list):
    partition_pt={}
    id2loc={}
    for n in node_list:
        shp=shape(MG.nodes[n]['geometry'])
        pt = shp

        if type(shp)==Polygon:
            pt = shp.centroid
        elif type(shp)==Point:
            pt = shp
        id2loc[n]=pt
        if (int(pt.x),int(pt.y)) not in partition_pt:
            partition_pt[(int(pt.x),int(pt.y))]=[]
        partition_pt[(int(pt.x),int(pt.y))].append(n)
    return partition_pt, id2loc

def find_nearest_in_G(Pt1, partition_pt, MG):
    min_d = 9999999999
    min_id_MG = -1
    for ii in [-1,0,1]:
        for jj in [-1,0,1]:
            if (int(Pt1.x)+ii,int(Pt1.y)+jj) not in partition_pt:
                continue
            for n0 in partition_pt[(int(Pt1.x)+ii,int(Pt1.y)+jj)]:
                Pt2 = get_node_pos(MG, n0)
                d = dist_btw_2Point(Pt1, Pt2)
                if d < min_d:
                    min_d = d
                    min_id_MG = n0
    return min_d, min_id_MG


def get_node_by_type_cc(MG, cc):
    type_set=set()
    node_by_type={}
    for n in cc:
        type_set.add(MG.nodes[n]['power'])
    node_by_type['plant']=[]
    node_by_type['hospital']=[]
    node_by_type['generator']=[]

    for t in type_set:
        node_by_type[t]=[]
        for n in cc:
            if MG.nodes[n]['power']==t:
                node_by_type[t].append(n)
    return node_by_type

def get_node_by_type_water(MG):
    type_set=set()
    node_by_type={}
    for n in MG.nodes:
        type_set.add(MG.nodes[n]['water'])
    for t in type_set:
        node_by_type[t]=[]
        for n in MG.nodes:
            if MG.nodes[n]['water']==t:
                node_by_type[t].append(n)
    return node_by_type

def get_node_by_type_communication(MG):
    type_set=set()
    node_by_type={}
    for n in MG.nodes:
        type_set.add(MG.nodes[n]['communication'])
    for t in type_set:
        node_by_type[t]=[]
        for n in MG.nodes:
            if MG.nodes[n]['communication']==t:
                node_by_type[t].append(n)
    return node_by_type