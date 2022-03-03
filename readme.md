# GPNet-simulator: evaluating robotic grasping in simulation

This repository provides the simulation used in GPNet as a more convenient Python package.
The original code can be found in the [GPNet repository](https://github.com/CZ-Wu/GPNet), 
which contains the Pytorch implementation of the GPNet paper:
[Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps](https://arxiv.org/abs/2009.12606).

The original code has only been modified to allow for better usability and unused files have been removed.

## Installation

```
cd GPNet-simulator
pip install -e .
```

Note that it is recommended to install the simulator in editable mode.
This way the simulator can find the object models and gripper models contained in the package.
If you install without editable mode, you will need to provide the correct paths yourself.

## Usage

Grasps are specified as a 6dof pose of position and quaternion. The simulation reads grasps from a text file with following structure:

```
shape1_name
x, y, z, rw, rx, ry, rz
x, y, z, rw, rx, ry, rz
shape2_name
x, y, z, rw, rx, ry, rz
...
```

Note that for the simulation to work properly, all grasps have to be moved along their z-axis 15mm
towards the object. This can be accomplished by using the option `--z_move`.
Per default this is switched off.

Simulation can be used either via CLI:

```
cd GPNet-simulator
python -m gpnet_sim -t gpnet_data/prediction/nms_poses_view0.txt
```

... or via API, as in `examples/sim_main.py`:

```
import os

import gpnet_sim

project_dir = os.path.join(os.path.dirname(__file__), '..')

conf = gpnet_sim.default_conf()
conf.testFile = os.path.join(project_dir, 'gpnet_data/prediction/nms_poses_view0.txt')
conf.z_move = True

top10, top30, top50, top100 = gpnet_sim.simulate(conf)
```

You can also use custom objects for the simulation. For this, provide the argument
`--objMeshRoot /path/to/urdf-files/`. For the simulation to work properly, the masses in kg should be multiplied by 
a factor of `0.001`.
See `gpnet_sim/simulator.py` for further arguments.

Some statistics will be written on the console, the full results will be stored to a csv log file (at same location as input file).
