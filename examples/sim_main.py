import os

import gpnet_sim

project_dir = os.path.join(os.path.dirname(__file__), '..')

conf = gpnet_sim.default_conf()
conf.testFile = os.path.join(project_dir, 'gpnet_data/prediction/nms_poses_view0.txt')
conf.z_move = True

top10, top30, top50, top100 = gpnet_sim.simulate(conf)

