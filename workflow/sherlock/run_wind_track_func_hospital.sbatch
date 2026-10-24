#!/bin/bash
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH -p serc

 
PROJECT_DIR="${INFRACASCADE_PROJECT_DIR:?set to the repository root}"
cd "$PROJECT_DIR"
ml python/3.9.0
export PYTHONPATH=$PROJECT_DIR/src/common:$PROJECT_DIR/src/networks:$PROJECT_DIR/src/hazard:$PROJECT_DIR/src/models:$PROJECT_DIR/src/analysis:$PROJECT_DIR/src/figures:/home/groups/jsuckale/zywei/envs/envs_osm/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/groups/jsuckale/zywei/envs/envs_osm/lib/
ml load json-glib/1.4.4
ml load py-scikit-learn/1.0.2_py39
ml load matlab/R2022b
ml load gcc/9.1.0

# python3 add_wind_to_power_network_for_track_func.py --track_id  "${1}"  --threshold  "${2}" --folder "${3}"
# python3 analyze_compare_flood_map.py --gdf_fn "${1}"
# python3 add_wind_to_power_network_for_track_func.py --track_id  "0"  --threshold  "25" --folder STORM_DATA_IBTRACS_SI_1000_YEARS_0


# python -m pip install matlabengine

# PYTHONUSERBASE=/home/groups/jsuckale/zywei/envs/envs_osm  pip install --user matlabengine

# PYTHONUSERBASE=/home/groups/jsuckale/zywei/envs/envs_osm  python3 -m pip install matlabengine==9.13.7
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/groups/jsuckale/zywei/envs/envs_osm/lib/
# ml load gcc/9.1.0

# python3 add_wind_to_power_network_for_track_func.py --track_id  0  --threshold  25 --folder STORM_DATA_IBTRACS_SI_1000_YEARS_2

python3 src/hazard/add_wind_to_hospital_for_track_func.py --threshold  25 --folder "${1}"
