import os
from time import time

import numpy as np

import gpnet_sim

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
MAX_GRASPS = 10000

if __name__ == '__main__':
    conf = gpnet_sim.default_conf()
    conf.z_move = True

    cpus = os.cpu_count()
    print('cpu count:', cpus)
    print('available cpus:', len(os.sched_getaffinity(0)))
    conf.processNum = cpus - 1

    grasps_fn = os.path.join(PROJECT_DIR, 'gpnet_data/prediction/some_grasps.npy')
    grasp_data = np.load(grasps_fn, allow_pickle=True)

    # there should actually only be one obj_id in the dict
    for obj_id in grasp_data.item().keys():
        centers = grasp_data.item()[obj_id]['centers'][:MAX_GRASPS]
        quats = grasp_data.item()[obj_id]['quaternions'][:MAX_GRASPS]
        print(f'number of grasps: {len(centers)}')
        print(f'number of processes: {conf.processNum}')
        start_time = time()
        gpnet_sim.simulate_direct(conf, obj_id, centers, quats)
        total_time = time() - start_time
        print(f'time required:\t\t{total_time} [s]')
        print(f'time required per process:\t{total_time/conf.processNum} [s]')
        print(f'sims per hr cpu time:\t{len(centers)/total_time*3600}')
