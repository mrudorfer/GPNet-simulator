import argparse
import os
import numpy as np
import quaternion
import tempfile
from attrdict import AttrDict

from . import AutoGraspShapeCoreUtil


def parser():
    parser = argparse.ArgumentParser(description='ShapeNetSem Grasp testing')
    parser.add_argument('-t', '--testFile',
                        default=os.path.join(os.path.dirname(__file__), '../gpnet_data/prediction/nms_poses_view0.txt'),
                        type=str, metavar='FILE', help='testFile path')
    parser.add_argument('-p', '--processNum', default=10, type=int, metavar='N', help='process num using')
    parser.add_argument('-w', "--width", action="store_true", dest="width", default=False,
                        help="turn on this param if test file contains width.")
    parser.add_argument('--gripperFile',
                        default=os.path.join(os.path.dirname(__file__), '../gpnet_data/gripper/parallel_simple.urdf'),
                        type=str, metavar='FILE', help='gripper file')
    parser.add_argument('--objMeshRoot', default=os.path.join(os.path.dirname(__file__), '../gpnet_data/urdf'),
                        type=str, metavar='PATH', help='obj mesh path')
    parser.add_argument('-v', '--visual', default=False, type=bool, metavar='VIS',
                        help='switch for visual inspection of grasps (processNum will be overridden)')
    parser.add_argument('-d', '--dir', default=None, type=str, metavar='PATH',
                        help='if this option is active, will search for test files in all subdirectories and compile' +
                             ' complete results file.')
    parser.add_argument('-z', '--z_move', default=False, type=bool,
                        help='if True, all grasp centers will be moved -15mm in their respective z-direction')
    parser.add_argument('--verbose', action='store_true', help='prints some output on console')

    return parser


def parse_args():
    return parser().parse_args()


def default_conf():
    return AttrDict(vars(parser().parse_args([])))


def getObjStatusAndAnnotation(testFile, haveWidth=False):
    with open(testFile, 'r') as testData:
        lines = testData.readlines()
        objIdList = []
        quaternionDict = {}
        centerDict = {}
        # 0: scaling    1~3: position   4~7: orientation    8: staticFrictionCoeff
        objId = 'invalid'
        objCounter = -1
        annotationCounter = -1
        for line in lines:
            # new object
            msg = line.strip()
            if len(msg.split(',')) < 2:
                objId = msg.strip()
                # skip invalid
                # begin
                objCounter += 1
                objIdList.append(objId)
                quaternionDict[objId] = np.empty(shape=(0, 4), dtype=float)
                centerDict[objId] = np.empty(shape=(0, 3), dtype=float)
                annotationCounter = -1
            # read annotation
            else:
                # skip invalid object
                if objId == 'invalid':
                    continue
                # begin
                annotationCounter += 1
                pose = msg.split(',')
                # print(objId, annotationCounter)
                if haveWidth:
                    length = float(pose[0]) * 0.085  # arbitrary value, will not be used in AutoGrasp
                    length = length if length < 0.085 else 0.085
                    position = np.array([float(pose[1]), float(pose[2]), float(pose[3])])
                    quaternion = np.array([float(pose[4]), float(pose[5]), float(pose[6]), float(pose[7])])
                    # quaternion = quaternion[[1, 2, 3, 0]]
                else:
                    length = 0.000  # arbitrary value, will not be used in AutoGrasp
                    position = np.array([float(pose[0]), float(pose[1]), float(pose[2])])
                    quaternion = np.array([float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])])
                    # quaternion = quaternion[[1, 2, 3, 0]]
                # print(objCounter, annotationCounter)
                quaternionDict[objId] = np.concatenate((quaternionDict[objId], quaternion[None, :]), axis=0)
                centerDict[objId] = np.concatenate((centerDict[objId], position[None, :]), axis=0)
    return quaternionDict, centerDict, objIdList


def find_test_files(directory):
    test_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            # print(os.path.join(subdir, file))
            if file == 'nms_poses_view0.txt':
                fn = os.path.join(subdir, file)
                print('found one:', fn)
                test_files.append(fn)
    return test_files


def z_move(c, q, z_move_length=0.015):
    """
    need to move by plus 15mm away from the center to make simulations
    :param c: (n, 3) ndarray translations
    :param q: (n, 4) ndarray quaternions (w, x, y, z)
    :param z_move_length: float, how much to move in each grasps z direction
    """
    rot_mats = quaternion.as_rotation_matrix(quaternion.from_float_array(q))
    offsets = rot_mats[:, :, 2] * z_move_length
    return c + offsets


def simulate(cfg):
    """
    Main simulator method.
    Pass this all described arguments as an attribute dictionary (where each item can be accessed like an attribute).
    See possible attributes above.

    :param cfg: an AttrDict
    :return: (4) top 10, top 30, top 50 and top 100 precision
    """
    if cfg.verbose:
        print('gpnet_simulator config:')
        if isinstance(cfg, argparse.Namespace):
            cfg = AttrDict(vars(cfg))
        for key, value in cfg.items():
            print(f'\t{key}:\t{value}')

    objMeshRoot = cfg.objMeshRoot
    processNum = cfg.processNum
    gripperFile = cfg.gripperFile
    haveWidth = cfg.width

    # check if directory-option has been used
    if cfg.dir is None or cfg.dir == 'None':
        testFiles = [cfg.testFile]
        dir_log_fn = None
    else:
        # cycle through directory to find all relevant files
        testFiles = find_test_files(cfg.dir)
        dir_log_fn = os.path.join(cfg.dir, 'simulation_log_file.txt')
        open(dir_log_fn, 'w').close()  # clear file

    visual = cfg.visual
    if visual:
        processNum = 1

    for testInfoFile in testFiles:
        if cfg.verbose:
            print('parsing test file: ', testInfoFile)
        logFile = testInfoFile[:-4] + '_log.csv'
        quaternionDict, centerDict, objIdList = getObjStatusAndAnnotation(testInfoFile, haveWidth)

        # print(f'objects: {objIdList}')
        # print(f'quaternions: {quaternionDict}')
        # print(f'centers: {centerDict}')

        open(logFile, 'w').close()
        simulator = AutoGraspShapeCoreUtil.AutoGraspUtil()

        for objId in objIdList:
            q = quaternionDict[objId]
            c = centerDict[objId]
            if cfg.z_move:
                # perform the z move
                c = z_move(c, q)
            simulator.addObject2(
                objId=objId,
                quaternion=q,
                translation=c
            )

        simulator.parallelSimulation(
            logFile=logFile,
            objMeshRoot=objMeshRoot,
            processNum=processNum,
            gripperFile=gripperFile,
            visual=visual
        )

        annotationSuccessDict = simulator.getSuccessData(logFile=logFile)
        top10, top30, top50, top100 = simulator.getStatistic(annotationSuccessDict)

        if cfg.verbose:
            print('results per object:')
            for key, arr in annotationSuccessDict.items():
                print(f'\t{key[:15] + "...":<18} success rate: {np.mean(arr)}')

            print('overall success rates:')
            print('\ttop10:\t', top10, '\n\ttop30:\t', top30, '\n\ttop50:\t', top50, '\n\ttop100:\t', top100)

        details, summary = simulator.get_simulation_summary(logFile)
        if cfg.verbose:
            print('absolute numbers by outcome:')
            for key, value in summary.items():
                print(f'\t{key}: {value}')

        if dir_log_fn is not None:
            with open(dir_log_fn, 'a') as log:
                log.write(f'FILE:\n{testInfoFile}\n')
                log.write(f'Simulation success rate at top k %%:\n')
                log.write(f'Top10\tTop30\tTop50\tTop100\n')
                log.write(f'{top10}\t{top30}\t{top50}\t{top100}\n')
                log.write(f'Simulation absolute results:\n')
                keys = [simulator.get_status_string(i) for i in range(7)]
                keys.append('total')
                [log.write(f'{key}\t') for key in keys]
                log.write('\n')
                [log.write(f'{summary[key]}\t') for key in keys]
                log.write('\n\n')

        return top10, top30, top50, top100


def simulate_direct(cfg, shape, centers, quats):
    """
    This is a direct simulation method that does not require writing things into a file.
    It does not produce a log file (although a temporary file is used) and is only capable of processing grasps
    for one specific object.

    :param cfg: a config file as in simulate method, except some attributes are not used
    :param shape: the object id
    :param centers: np array with grasp centers
    :param quats: np array with grasp quaternions

    :return: binary success array, dict with error types
    """
    print('gpnet_simulator config:')
    if isinstance(cfg, argparse.Namespace):
        cfg = AttrDict(vars(cfg))
    for key, value in cfg.items():
        print(f'\t{key}:\t{value}')

    objMeshRoot = cfg.objMeshRoot
    processNum = cfg.processNum
    gripperFile = cfg.gripperFile

    # this creates a logfile in the tmp dir of the operating system
    logFileHandle, logFile = tempfile.mkstemp(suffix='.log', text=True)

    open(logFile, 'w').close()
    simulator = AutoGraspShapeCoreUtil.AutoGraspUtil()

    if cfg.z_move:
        centers = z_move(centers, quats)

    simulator.addObject2(
        objId=shape,
        quaternion=quats,
        translation=centers
    )

    simulator.parallelSimulation(
        logFile=logFile,
        objMeshRoot=objMeshRoot,
        processNum=processNum,
        gripperFile=gripperFile
    )

    result_dict = AutoGraspShapeCoreUtil.read_sim_csv_file(logFile, initial_array_size=len(centers))
    grasps = result_dict[shape]
    sim_outcome = grasps[:, 8]
    sim_success = grasps[:, 9]

    if cfg.verbose:
        print('simulation results for shape', shape)
        print(f'\tsuccess rate: {np.mean(sim_success)}')

    status_code, counts = np.unique(sim_outcome, return_counts=True)
    summary = {}
    for i in range(len(status_code)):
        summary[simulator.get_status_string(status_code[i])] = counts[i]

    if cfg.verbose:
        print('absolute numbers by outcome:')
        for key, value in summary.items():
            print(f'\t{key}: {value}')

    os.close(logFileHandle)
    os.remove(logFile)

    return sim_success, summary


if __name__ == "__main__":
    args = parse_args()
    simulate(cfg=args)
