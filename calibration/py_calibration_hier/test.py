from submodel import SubModelFP
from transport import TransportFP

paths_tc = {
    'path_x' : './copper/inputs_sim_tc.csv',
    'path_y' : './copper/outputs_sim_tc.csv',
    'path_y_actual' : './copper/outputs_real_tc.csv',
    }
paths_fp = {
    'path_x' : './copper/inputs_sim_fp.txt',
    'path_y' : './copper/outputs_sim_fp.csv',
    'path_y_actual' : './copper/outputs_real_fp.csv'
    }

sm = SubModelFP(TransportFP(**paths_fp))

xtry = sm.X.mean(axis = 0)


# EOF
