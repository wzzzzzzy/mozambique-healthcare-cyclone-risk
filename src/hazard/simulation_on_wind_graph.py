import argparse
import os
import pickle
import networkx as nx
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import warnings
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from power_capacity import (
    hospital_capacity_failure_probability,
    legacy_capacity_failure_probability,
)

warnings.filterwarnings('ignore')

CAPACITY_MODELS = {'legacy', 'unmet_demand'}


def draw_edge_random_values(
    df_edge,
    shared_edge_rng=None,
    ablation_edge_rng=None,
):
    """Draw edge samples without shifting the shared-edge random stream."""

    if shared_edge_rng is None:
        return np.random.uniform(0, 1, len(df_edge))
    if ablation_edge_rng is None:
        raise ValueError('An ablation-edge random stream is required')

    ablation_mask = df_edge['ablation'].to_numpy(dtype=bool)
    draws = np.empty(len(df_edge), dtype=float)
    draws[~ablation_mask] = shared_edge_rng.uniform(
        0, 1, np.count_nonzero(~ablation_mask)
    )
    draws[ablation_mask] = ablation_edge_rng.uniform(
        0, 1, np.count_nonzero(ablation_mask)
    )
    return draws


def generate_df_edge(simplified_hospital_MG_wind):
    df_edge = pd.DataFrame()
    ids = []
    line_fails = []
    tower_fails = []
    ablation_flags = []
    regular_edges = []
    ablation_edges = []
    for e in simplified_hospital_MG_wind.edges:
        if simplified_hospital_MG_wind.edges[e]['line_fail']>0:
            edge_data = simplified_hospital_MG_wind.edges[e]
            target = ablation_edges if edge_data.get('ablation') else regular_edges
            target.append(e)
    # Keep all shared edges first and in their original order so a public run
    # and its all-grid ablation use the same random draws for those edges.
    for e in tqdm(regular_edges + ablation_edges):
        ids.append(e)
        line_fails.append(simplified_hospital_MG_wind.edges[e]['line_fail'])
        tower_fails.append(simplified_hospital_MG_wind.edges[e]['tower_fail'])
        ablation_flags.append(
            bool(simplified_hospital_MG_wind.edges[e].get('ablation'))
        )
    df_edge['id'] = ids
    df_edge['line_fail'] = line_fails
    df_edge['tower_fail'] = tower_fails
    df_edge['ablation'] = ablation_flags
    return df_edge

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

def network_original(MG_hospital, s2mw):
    ccs = []
    len_ccs = []
    supplies = []
    unknowns = []
    num_hospitals = []
    hospitals = []

    for cc in list(nx.connected_components(MG_hospital)):
        # print(len(cc))
        ccs.append(list(cc))
        len_ccs.append(len(list(cc)))
        node_by_type_cc = get_node_by_type_cc(MG_hospital, list(cc))
        supply = 0
        unknown = []
        for n in node_by_type_cc['plant']:
            if s2mw[n] != -1:
                supply+= s2mw[n]
            else:
                unknown.append(n)
        for n in node_by_type_cc['generator']:
            unknown.append(n)
        num_hospital = len(node_by_type_cc['hospital'])
        supplies.append(supply)
        unknowns.append(unknown)
        num_hospitals.append(num_hospital)
        hospitals.append(node_by_type_cc['hospital'])


    df =pd.DataFrame()
    df['ccs']=ccs
    df['len_ccs']=len_ccs
    df['supplies']=supplies
    df['unknowns']=unknowns
    df['num_hospitals']=num_hospitals
    df['hospitals']=hospitals
    return df

def network_break_down(
    MG_hospital,
    tag,
    s2mw,
    remove_list,
    threshold,
    folder=0,
    track_id=0,
    capacity_model='legacy',
):
    if capacity_model not in CAPACITY_MODELS:
        raise ValueError(
            f'capacity_model must be one of {sorted(CAPACITY_MODELS)}, '
            f'not {capacity_model!r}'
        )
    all_hosp = []
    MG_hospital_removes = MG_hospital.copy()
    MG_hospital_removes.remove_edges_from(remove_list)#df_edges_fail_assure['id'].tolist()
    not_transmission = ['MG-hospital','gridfinder','OSM-gridfinder','connect','connect-isolated']

    ccs = []
    len_ccs = []
    supplies = []
    unknowns = []
    num_hospitals = []
    hospitals = []
    fail_probs = []
    dict_n2fail_probs = {}
    fail_transmission = 0
    for e in remove_list:
        if MG_hospital.edges[e]['attribute']['power'] not in not_transmission:
            fail_transmission += MG_hospital.edges[e]['length']
    initial_fail = fail_transmission
    for cc in list(nx.connected_components(MG_hospital_removes)):
        # print(len(cc))
        ccs.append(list(cc))
        len_ccs.append(len(list(cc)))
        node_by_type_cc = get_node_by_type_cc(MG_hospital_removes, list(cc))
        supply = 0
        unknown = []
        for n in node_by_type_cc['plant']:
            if s2mw[n] != -1:
                supply+= s2mw[n]
            else:
                # unknown.append(n)
                supply+= 80.0
        for n in node_by_type_cc['generator']:
            unknown.append(n)
        num_hospital = len(node_by_type_cc['hospital'])
        supplies.append(supply)
        unknowns.append(unknown)
        num_hospitals.append(num_hospital)
        hospitals.append(node_by_type_cc['hospital'])
        f_p = 0
        subgraph = MG_hospital_removes.subgraph(cc)
        if supply == 0 and len(unknown)==0:
            all_hosp.extend(node_by_type_cc['hospital'])
            # fail_probs.append(1)
            f_p = 1
            for e in subgraph.edges:
                if MG_hospital.edges[e]['attribute']['power'] not in not_transmission:
                    fail_transmission += subgraph.edges[e]['length']
        elif num_hospital==0 or supply/num_hospital > 4 or len(unknown)>0:
            # fail_probs.append(0)
            f_p = 0
        else:
            if capacity_model == 'legacy':
                f_p = legacy_capacity_failure_probability(
                    supply, num_hospital
                )
            else:
                # Four MW per hospital is treated as adequate supply. Failure
                # decreases as available supply approaches that level.
                f_p = hospital_capacity_failure_probability(
                    supply, num_hospital
                )
        fail_probs.append(f_p)
        for n in cc:
            dict_n2fail_probs[n]=f_p


    df = pd.DataFrame()
    df['ccs']=ccs
    df['len_ccs']=len_ccs
    df['supplies']=supplies
    df['unknowns']=unknowns
    df['num_hospitals']=num_hospitals
    df['hospitals']=hospitals
    df['fail_prob']=fail_probs
    component_result_dir = (
        '/scratch/users/zywei/cascade_result/cc_csv_cyclone'
    )
    os.makedirs(component_result_dir, exist_ok=True)
    df.to_csv(
        f'{component_result_dir}/{folder}_{track_id}_{tag}_{threshold}.csv'
    )

    return df, all_hosp, dict_n2fail_probs, fail_transmission, initial_fail



def monte_carlo_simulation_till_converge(
    simplified_hospital_MG_wind,
    public_net,
    df_edge,
    s2mw,
    threshold,
    folder=0,
    track_id=0,
    capacity_model='legacy',
    random_seed=None,
):
    iter = 0
    public_net = public_net[public_net['geometry']!=None]
    curr = np.zeros(len(public_net))
    fail_transmission_list = []
    initial_fail_list = []
    if random_seed is not None:
        shared_edge_rng = np.random.RandomState(random_seed)
        ablation_edge_rng = np.random.RandomState(random_seed + 1)
    while iter<10000:
        iter+=1
        r = draw_edge_random_values(
            df_edge,
            shared_edge_rng if random_seed is not None else None,
            ablation_edge_rng if random_seed is not None else None,
        )
        rm_list = df_edge[(df_edge['line_fail']>r)]['id'].tolist()
        _, _, dict_n2fail_probs,fail_transmission, initial_fail = network_break_down(
            simplified_hospital_MG_wind,
            iter,
            s2mw,
            rm_list,
            threshold,
            folder,
            track_id,
            capacity_model,
        )
        fail_transmission_list.append(fail_transmission)
        initial_fail_list.append(initial_fail)
        # print(fail_transmission)
        f_ps=[]
        for i in public_net.index:
            id = public_net['node_id'][i]
            if id in dict_n2fail_probs:
                f_ps.append(dict_n2fail_probs[id])
            else:
                f_ps.append(-1)
        public_net[f'fp_{iter}']= f_ps

        if iter % 100==0:
            last = curr.copy()
            # public_net_tmp= public_net[public_net['geometry']!=None]
            curr = public_net.iloc[:,-iter:].to_numpy().mean(axis=1)
            diff = abs(last-curr)
            # print(iter,sum(diff))
            if (diff<0.01).all():
                public_net[f'final']= curr
                print(threshold, sum(curr)/len(curr),len(curr), sum(curr>0),sum(fail_transmission_list)/len(fail_transmission_list), sum(initial_fail_list)/len(initial_fail_list))
                tmp = {"threshold":threshold,
                       "avg_hospital": sum(curr)/len(curr),
                       "num_hopsital": len(curr),
                       "failed_hopsital":  sum(curr>0),
                       "fail_transmission": sum(fail_transmission_list)/len(fail_transmission_list),
                       "initial_fail": sum(initial_fail_list)/len(initial_fail_list),
                       "iterations": iter,
                       "capacity_model": capacity_model,
                       "random_seed": random_seed}
                failure_result_dir = '/scratch/users/zywei/cascade_result/failure'
                os.makedirs(failure_result_dir, exist_ok=True)
                with open(f'{failure_result_dir}/{threshold}_{folder}_{track_id}.pickle', 'wb') as handle:
                    pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)


                break
    return public_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--tiff_name',  type=str)
    parser.add_argument('--threshold',  type=int, default=24)
    args = parser.parse_args()
    tiff_name = args.tiff_name
    threshold = args.threshold
    print('threshold',threshold)
    with open(f'network_save/{tiff_name}_wind_{threshold}.pickle', 'rb') as handle:
        simplified_hospital_MG_wind = pickle.load(handle)
    df_edge = generate_df_edge(simplified_hospital_MG_wind)

    with open('network_save/s2mw.pickle', 'rb') as handle:
        s2mw = pickle.load(handle)

    df_orig = network_original(simplified_hospital_MG_wind, s2mw)

    hospital = gpd.read_file('SARA_processed')
    public_net = hospital[hospital['ele_main_s']==1]
    nodes_list = []
    for  i in tqdm(public_net.index):
        nodes_list.append(f'node/46774825{str(i).zfill(4)}')
    public_net['node_id'] = nodes_list

    public_net = monte_carlo_simulation_till_converge(simplified_hospital_MG_wind, public_net, df_edge, s2mw, threshold)
    public_net.to_file(f'/scratch/users/zywei/wind_result/{tiff_name}_{threshold}')

    fig, base = plt.subplots(figsize=(8, 8))

    public_net.plot(ax=base, column="final", s = 40, vmin=0, vmax = 1, edgecolors='k', legend=True)
    base.set_xlim(30, 41)
    base.set_ylim(-27, -10)
    base.set_xticks([])
    base.set_yticks([])
    plt.savefig(f'/scratch/users/zywei/wind_result/fig/{tiff_name}_{threshold}.png')
