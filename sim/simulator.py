import argparse
import os
import numpy as np
from simulateTest.AutoGraspShapeCoreUtil import AutoGraspUtil


# TODO: Check modification Martin if fits test_all.py and eval_all.py
def parse_args():
    # parser = argparse.ArgumentParser(description='ShapeNetSem Grasp testing')
    # parser.add_argument('-t', '--testFile', default='prediction/500ntop10.txt', type=str, metavar='FILE', help='testFile path')
    # # parser.add_argument('-n', '--samplePerObject', default=10, type=int, metavar='N', help='sample num per object')
    # parser.add_argument('-p', '--processNum', default=18, type=int, metavar='N', help='process num using')
    # parser.add_argument('-w', '--haveWidth', default=1, type=int, metavar='N', help='0 : no width ; 1 : have width')
    # parser.add_argument('--gripperFile', default='/data/shapeNet/annotator2/parallel_simple.urdf', type=str, metavar='FILE', help='gripper file')
    # parser.add_argument('--objMeshRoot', default='/data/shapeNet/urdf', type=str, metavar='PATH', help='obj mesh path')

    parser = argparse.ArgumentParser(description='ShapeNetSem Grasp testing')
    parser.add_argument('-t', '--testFile', default='gpnet_data/prediction/nms_poses_view0.txt', type=str, metavar='FILE',
                        help='testFile path')
    # parser.add_argument('-n', '--samplePerObject', default=10, type=int, metavar='N', help='sample num per object')
    parser.add_argument('-p', '--processNum', default=10, type=int, metavar='N', help='process num using')
    # parser.add_argument('-w', '--haveWidth', default=0, type=int, metavar='N', help='0 : no width ; 1 : have width')
    parser.add_argument('-w', "--width", action="store_true", dest="width", default=False,
                        help="turn on this param if test file contains width.")
    parser.add_argument('--gripperFile', default='gpnet_data/gripper/parallel_simple.urdf', type=str, metavar='FILE',
                        help='gripper file')
    parser.add_argument('--objMeshRoot', default='gpnet_data/urdf', type=str, metavar='PATH', help='obj mesh path')
    parser.add_argument('-v', '--visual', default=False, type=bool, metavar='VIS',
                        help='switch for visual inspection of grasps (no parallelisation, processNum will be overridden)')
    parser.add_argument('-d', '--dir', default='None', type=str, metavar='PATH',
                        help='if this option is active, will search for test files in all subdirectories and compile a' +
                        ' complete results file.')
    parser.add_argument('-l', '--limit', default=None, type=int, metavar='LIMIT',
                    help='if set, NMS will be disabled and the provided number of predictions will' +
                         ' be simulated per object')
    parser.add_argument('-z', '--z_move', default=False, type=bool,
                        help='if True, all grasp centers will be moved -15mm in their respective z-direction')
    return parser.parse_args()



# TODO: Martin?
def getObjStatusAndAnnotation_fromNPZ(npz_dir, limit=None):
    obj_id_list = []
    quaternion_dict = {}
    center_dict = {}

    for subdir, dirs, files in os.walk(npz_dir):
        for file in files:
            if file[-4:] == '.npz':
                fn = os.path.join(subdir, file)
                print('parsing', fn)
                shape = file[:-4]
                with np.load(fn) as data:
                    # sort by score
                    order = np.argsort(-data['scores'])
                    center_dict[shape] = data['centers'][order][:limit]
                    quaternion_dict[shape] = data['quaternions'][order][:limit]
                    obj_id_list.append(shape)

    return quaternion_dict, center_dict, obj_id_list


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
                quaternionDict[objId] = np.empty(shape=(0, 4), dtype=np.float)
                centerDict[objId] = np.empty(shape=(0, 3), dtype=np.float)
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
    :param c: (n, 3)
    :param q: (n, 4)
    :param z_move_length: float, how much to move in each grasps z direction
    """
    import burg_toolkit as burg
    graspset = burg.grasp.GraspSet.from_translations_and_quaternions(c, q)
    offsets = graspset.rotation_matrices[:, :, 2] * z_move_length
    return graspset.translations + offsets


def main_simulator(cfg):
    objMeshRoot = cfg.objMeshRoot
    processNum = cfg.processNum
    gripperFile = cfg.gripperFile
    haveWidth = cfg.width
    limit = cfg.limit
    use_all_grasps = False if limit is None else True

    # check if directory-option has been used
    if cfg.dir == 'None':
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
        if use_all_grasps:
            logFile = os.path.join(os.path.dirname(testInfoFile), f'view0_{limit}_log.csv')
            quaternionDict, centerDict, objIdList = getObjStatusAndAnnotation_fromNPZ(
                os.path.join(os.path.dirname(testInfoFile), 'view0/'), limit=limit)
        else:
            logFile = testInfoFile[:-4] + '_log.csv'
            quaternionDict, centerDict, objIdList = getObjStatusAndAnnotation(testInfoFile, haveWidth)

        # print(f'objects: {objIdList}')
        # print(f'quaternions: {quaternionDict}')
        # print(f'centers: {centerDict}')

        open(logFile, 'w').close()
        simulator = AutoGraspUtil()

        for objId in objIdList:
            q = quaternionDict[objId]
            c = centerDict[objId]
            if cfg.z_move:
                # perform the z move
                c = z_move(c, q)
                print('moving z for simulation')
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
        # print(top 10% 30% 50% 100%)
        print(annotationSuccessDict)
        top10, top30, top50, top100 = simulator.getStatistic(annotationSuccessDict)
        print('top10:\t', top10, '\ntop30:\t', top30, '\ntop50:\t', top50, '\ntop100:\t', top100)

        details, summary = simulator.get_simulation_summary(logFile)
        print(summary)

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

        return (top10, top30, top50, top100)


if __name__ == "__main__":
    args = parse_args()
    main_simulator(cfg=args)
