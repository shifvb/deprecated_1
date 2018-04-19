from glob import glob

def transport(inp,op):
    '''
    数据转换
    :param inp: 所有病人所在的文件夹
    :param op: 要输出的目标文件夹
    :return:
    '''
    tarPart='xpdata\DICOMIMG'
    import os
    import shutil
    import pydicom as dicom
    if not (os.path.exists(op)):
        os.mkdir(op)
    patList=os.listdir(inp)
    for patName in patList:
        fileDirPath=os.path.join(inp,patName,tarPart)
        ctdir=glob(os.path.join(fileDirPath,'ct','*'))
        petdir = glob(os.path.join(fileDirPath, 'pet','*'))
        # fileDir=os.listdir(fileDirPath)
        fileDir = ctdir+petdir
        if not os.path.exists(os.path.join(op, patName)):
            os.mkdir(os.path.join(op, patName))
        ctPath = os.path.join(op, patName, 'CT')
        if not os.path.exists(ctPath):
            os.mkdir(ctPath)
        petPath = os.path.join(op, patName, 'PT')
        if not os.path.exists(petPath):
            os.mkdir(petPath)
        for fileName in fileDir:
            oldPath=os.path.join(fileDirPath,fileName)
            oldPath=fileName
            meta=dicom.read_file(oldPath)
            instanceNumber=meta.get('InstanceNumber')
            modality=meta.get('Modality')
            if instanceNumber<10:
                newName=modality+'_00'+str(instanceNumber)
            elif instanceNumber<100:
                newName = modality + '_0' + str(instanceNumber)
            else:
                newName = modality  + '_' +str(instanceNumber)
            newPath=os.path.join(op,patName,modality,newName)
            shutil.copy(oldPath,newPath)
if __name__ == '__main__':
    #将从医院拷来的原始数据 去掉无用的相关数据，并整理好格式
    inp = r'F:\registration\3'
    op = r'F:\registration\3o'
    transport(inp,op)
