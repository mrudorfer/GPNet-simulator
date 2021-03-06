import gc
import os
import csv

import numpy as np
import pybullet
from joblib import Parallel, delayed
from tqdm import tqdm

from .AutoGraspSimpleShapeCore import AutoGraspSimple


class AutoGraspUtil(object):
    def __init__(self):
        self.__memoryInit()

    def __memoryInit(self):
        self.objIdList = []
        self.annotationDict = {}

    def __annotationMemoryReallocate(self):
        del self.annotationDict
        gc.collect()
        self.annotationDict = {}


    # objId: str (len <= 32)
    # quaternion: ndarray [N, 4]
    # translation: ndarray [N, 3]
    def addObject2(self, objId, quaternion, translation):
        # add object
        self.objIdList.append(objId)

        # add annotation
        annotationNum = quaternion.shape[0]
        # fit the pybullet environment
        quaternion = quaternion[:, [1, 2, 3, 0]]
        fakeLength = np.zeros((annotationNum, 1))
        annotation = np.concatenate((fakeLength, translation, quaternion), axis=1)
        self.annotationDict[objId] = annotation

    # used to 
    def addObjectSplit(self, objId, quaternion, translation, logFile, objMeshRoot, processNum, gripperFile,
                       splitLen=50):
        # add object
        self.objIdList.append(objId)
        objNum = len(self.objIdList)
        if objNum <= 1:
            # erase log file
            open(logFile, 'w').close()

        # add annotation
        annotationNum = quaternion.shape[0]
        # fit the pybullet environment
        quaternion = quaternion[:, [1, 2, 3, 0]]
        fakeLength = np.zeros((annotationNum, 1))
        annotation = np.concatenate((fakeLength, translation, quaternion), axis=1)
        self.annotationDict[objId] = annotation
        if (objNum % splitLen) == 0:
            splitIndex = objNum // splitLen
            runningObjIdList, \
            runningObjIndexList, \
            runningAnnotaionList, \
            runningAnnotaionIndexList = self.__getRunningListSplit(
                objIdListSplit=self.objIdList[(splitIndex - 1) * splitLen: splitIndex * splitLen],
                objIndexShift=(splitIndex - 1) * splitLen
            )
            objMeshRoot = objMeshRoot
            with Parallel(n_jobs=processNum, backend='multiprocessing') as parallel:
                parallel(delayed(AutoGraspUtil.testAnnotation)
                         (objId, objIndex, annotation, annotationIndex, gripperFile, logFile, objMeshRoot)
                         for (objId, objIndex, annotation, annotationIndex) in
                         zip(runningObjIdList, runningObjIndexList, runningAnnotaionList, runningAnnotaionIndexList))
            self.__annotationMemoryReallocate()

    def __getRunningListSplit(self, objIdListSplit, objIndexShift):
        runningObjIdList = []
        runningObjIndexList = []
        runningAnnotaionList = []
        runningAnnotaionIndexList = []

        for objIndex, objId in enumerate(objIdListSplit):
            for annotationIndex, annotation in enumerate(self.annotationDict[objId]):
                runningObjIdList.append(objId)
                runningObjIndexList.append(objIndex + objIndexShift)
                runningAnnotaionList.append(annotation)
                runningAnnotaionIndexList.append(annotationIndex)

        return runningObjIdList, runningObjIndexList, runningAnnotaionList, \
               runningAnnotaionIndexList

    def parallelSimulation(self, logFile, objMeshRoot, processNum, gripperFile, visual=False):
        runningObjIdList, \
        runningObjIndexList, \
        runningAnnotaionList, \
        runningAnnotaionIndexList = self.__getRunningList()
        # erase log file
        open(logFile, 'w').close()
        objMeshRoot = objMeshRoot
        print('starting simulation...')
        with Parallel(n_jobs=processNum, backend='multiprocessing') as parallel:
            parallel(delayed(AutoGraspUtil.testAnnotation)
                     (objId, objIndex, annotation, annotationIndex, gripperFile, logFile, objMeshRoot, visual)
                     for (objId, objIndex, annotation, annotationIndex) in
                     tqdm(zip(runningObjIdList, runningObjIndexList, runningAnnotaionList, runningAnnotaionIndexList),
                          total=len(runningObjIdList))
                     )

    def __getRunningList(self):
        # annotationNum = annotationDict.shape[1]
        runningObjIdList = []
        runningObjIndexList = []
        runningAnnotaionList = []
        runningAnnotaionIndexList = []
        for objIndex, objId in enumerate(self.objIdList):
            for annotationIndex, annotation in enumerate(self.annotationDict[objId]):
                runningObjIdList.append(objId)
                runningObjIndexList.append(objIndex)
                runningAnnotaionList.append(annotation)
                runningAnnotaionIndexList.append(annotationIndex)

        return runningObjIdList, runningObjIndexList, runningAnnotaionList, \
               runningAnnotaionIndexList

    @staticmethod
    def testAnnotation(objId, objIndex, annotation, annotationIndex, gripperFile, logfile,
                       objMeshRoot, visual=False):
        status = AutoGraspUtil.annotationSimulation(
            objId=objId,
            annotation=annotation,
            objMeshRoot=objMeshRoot,
            gripperFile=gripperFile,
            visual=visual,
        )
        #print('objId\t', objId, '\tobjIdx\t', str(objIndex), '\tannotationIdx\t', str(annotationIndex), '\tstatus\t',
        #      str(status))
        # todo: progress bar ?
        simulatorParamStrList = [str(i) for i in annotation]
        logInfo = [objId, str(annotationIndex), str(status)]
        simulatorParamStrList = logInfo + simulatorParamStrList
        # print(simulatorParamStrList)
        simulatorParamStr = ','.join(simulatorParamStrList)
        # logLine = objId + ',' + str(annotationIndex) + ',' + str(status) + '\n'
        logLine = simulatorParamStr + '\n'
        # print(logLine)
        with open(logfile, 'a') as logger:
            logger.write(logLine)

    @staticmethod
    def getSuccessData(logFile):
        annotationSuccessDict = {}
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                # objId, annotationId, status = msg.split(',')
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])
                # initialize
                if objId not in annotationSuccessDict.keys():
                    annotationSuccessDict[objId] = np.full(shape=[annotationId + 1], fill_value=False)
                # # if success
                # if status == 0:
                #     logDict[objId][annotationId, gripperId] = True

                # if overflow (keep rank): reallocate memory
                if annotationId + 1 > len(annotationSuccessDict[objId]):
                    temp = np.full(shape=[annotationId + 1], fill_value=False)
                    temp[:len(annotationSuccessDict[objId])] = annotationSuccessDict[objId]
                    annotationSuccessDict[objId] = temp

                # if success
                if status == 0:
                    annotationSuccessDict[objId][annotationId] = True

        return annotationSuccessDict

    @staticmethod
    def getCollisionData(logFile):
        annotationSuccessDict = {}
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])

                # initialize
                if objId not in annotationSuccessDict.keys():
                    annotationSuccessDict[objId] = np.full(shape=[annotationId + 1], fill_value=False)

                # # if success
                # if status == 0:
                #     logDict[objId][annotationId, gripperId] = True

                # if overflow (keep rank): reallocate memory
                if annotationId + 1 > len(annotationSuccessDict[objId]):
                    temp = np.full(shape=[annotationId + 1], fill_value=False)
                    temp[:len(annotationSuccessDict[objId])] = annotationSuccessDict[objId]
                    annotationSuccessDict[objId] = temp

                # if success
                if status == 2 or status == 1:
                    annotationSuccessDict[objId][annotationId] = True

        return annotationSuccessDict

    @staticmethod
    def annotationVisualization(logFile, objIdv, annotationIdv, objMeshRoot, gripperFile):
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                # objId, annotationId, status = msg.split(',')
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])
                simulatorParamStr = msgList[3:]
                simulatorParam = [float(i) for i in simulatorParamStr]
                annotation = np.array(simulatorParam[:8])
                if objId == objIdv and annotationId == annotationIdv:
                    status = AutoGraspUtil.annotationSimulation(
                        objId=objId,
                        annotation=annotation,
                        objMeshRoot=objMeshRoot,
                        gripperFile=gripperFile,
                        visual=True
                    )
                    return status

    @staticmethod
    def annotationSimulation(objId, annotation, objMeshRoot, gripperFile, visual=False):
        length = annotation[0]
        position = annotation[1:4]
        quaternion = annotation[4:8]
        autoGraspInstance = AutoGraspSimple(
            # clientId = gripperIndex,
            objectURDFFile=os.path.join(objMeshRoot, objId + ".urdf"),
            gripperLengthInit=length,
            gripperBasePosition=position,
            gripperURDFFile=gripperFile,
            gripperBaseOrientation=quaternion,
            serverMode=pybullet.GUI if visual else pybullet.DIRECT,
            # serverMode=pybullet.GUI,
        )
        result = autoGraspInstance.startSimulation()
        if visual:
            print(f'simulation result: {AutoGraspUtil.get_status_string(result)} (press enter)')
            input()
        return result

    @staticmethod
    def getStatistic(annotationSuccessDict):
        currentObjNum = len(annotationSuccessDict.keys())
        top10Success = np.empty(shape=currentObjNum, dtype=np.float)
        top30Success = np.empty(shape=currentObjNum, dtype=np.float)
        top50Success = np.empty(shape=currentObjNum, dtype=np.float)
        top100Success = np.empty(shape=currentObjNum, dtype=np.float)

        for index, annotationSuccess in enumerate(annotationSuccessDict.values()):
            annotationNum = len(annotationSuccess)
            # print(annotationSuccess)
            top10_num = int(0.1 * annotationNum)
            if top10_num == 0:
                top10_num = 1
            top10Success[index] = np.sum(
                annotationSuccess[:top10_num]) / top10_num
            top30_num = round(0.3 * annotationNum)
            if top30_num == 0:
                top30_num = 1
            top30Success[index] = np.sum(
                annotationSuccess[:top30_num]) / top30_num
            top50_num = round(0.5 * annotationNum)
            if top50_num == 0:
                top50_num = 1
            top50Success[index] = np.sum(
                annotationSuccess[:top50_num]) / top50_num
            top100Success[index] = np.sum(annotationSuccess) / annotationNum
        return top10Success.mean(), top30Success.mean(), top50Success.mean(), top100Success.mean()

    @staticmethod
    def listSplit(listToSplit, splitLen=48000 * 19 * 50):
        return [listToSplit[splitLen * i: splitLen * (i + 1)] for i in range(len(listToSplit) // splitLen + 1)]

    @staticmethod
    def get_simulation_summary(logFile):
        annotationSuccessDict = {}
        status_frequencies = np.zeros(7, dtype=np.int)
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])

                # initialize
                if objId not in annotationSuccessDict.keys():
                    annotationSuccessDict[objId] = []

                # append status to obj
                annotationSuccessDict[objId].append(AutoGraspUtil.get_status_string(status))
                status_frequencies[status] += 1

        freq_dict = {}
        for status, freq in enumerate(status_frequencies):
            freq_dict[AutoGraspUtil.get_status_string(status)] = freq

        freq_dict['total'] = status_frequencies.sum()

        return annotationSuccessDict, freq_dict

    @staticmethod
    def get_status_string(status):
        if status == 0:
            return 'success'
        if status == 1:
            return 'collision with ground'
        if status == 2:
            return 'collision with object'
        if status == 3:
            return 'object untouched'
        if status == 4:
            return 'incorrect contact'
        if status == 5:
            return 'object fallen'
        if status == 6:
            return 'time out'
        else:
            return 'unknown status'


def read_sim_csv_file(filename, keep_num=None, initial_array_size=2000):
    """
    This reads the csv log file created during simulation.

    :param filename: the filename of the simulation's log file output
    :param keep_num: at most this number of grasps is reported (as of annotation idx order)
    :param initial_array_size: an estimate of how many grasps there will be per object to speed up things

    :return: returns a dict with shape id as keys and np array as value.
             the np array is of shape (n, 10): 0:3 pos, 3:7 quat, annotation id, sim result, sim success
             keeps only keep_num entries (as of annotation idx order). quaternion is in w,x,y,z
    """

    print(f'reading csv data from {filename}')
    sim_data = {}
    counters = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(reader):
            shape = row[0]
            if shape not in sim_data.keys():
                # we do not know the array length in advance, so start with 10k
                data_array = np.zeros((initial_array_size, 10))
                sim_data[shape] = data_array
                counters[shape] = 0
            elif counters[shape] == len(sim_data[shape]):
                sim_data[shape] = np.resize(sim_data[shape], (len(sim_data[shape]) + initial_array_size, 10))

            sim_data[shape][counters[shape]] = [
                float(row[4]),  # pos: x, y, z
                float(row[5]),
                float(row[6]),
                float(row[10]),  # quat: w, x, y, z, converted from pybullet convention
                float(row[7]),
                float(row[8]),
                float(row[9]),
                int(row[1]),  # annotation id
                int(row[2]),  # simulation result
                int(row[2]) == 0  # simulation success flag
            ]
            counters[shape] += 1

    # now reduce arrays to their actual content
    for key in sim_data.keys():
        sim_data[key] = np.resize(sim_data[key], (counters[key], 10))
        # also sort by annotation id
        order = np.argsort(sim_data[key][:, 7])
        sim_data[key] = sim_data[key][order]
        sim_data[key] = sim_data[key][:keep_num]

    return sim_data
