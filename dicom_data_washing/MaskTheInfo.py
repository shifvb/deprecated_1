'''
该工具用于抹去患者信息，包括id，身高，体重，生日，姓名，年龄性别等
与patientId对应的id在保留的testp.txt文件中可以查到,该版本是另一版
'''
'''
edit 2018/03/16 修改，保留了患者的身高体重和性别、年龄信息
'''
import os
import csv
import pydicom as pdm
import glob
import natsort
import shutil

ctDir='CT'
ptDir='PT'

# ctDir = '4'
# ptDir = '5'


def maskTheInfo(inp, opRoot):
    if not os.path.exists(opRoot):
        os.mkdir(opRoot)

    patientNames = os.listdir(inp)
    index = 0
    menu = []
    for patientFileId in patientNames:
        dirCtPath = os.path.join(inp, patientFileId, ctDir)
        fileCtNames = os.listdir(dirCtPath)
        # patientDirPath = os.path.join(opRoot, str(index))

        for fileCtName in fileCtNames:
            if fileCtName.startswith('CT_'):
                meta = pdm.read_file(os.path.join(dirCtPath, fileCtName))
                id = meta.PatientID

                patientDirPath = os.path.join(opRoot, id)
                if not os.path.exists(patientDirPath):
                    os.mkdir(patientDirPath)
                opCtDir = os.path.join(patientDirPath, 'CT')
                if not os.path.exists(opCtDir):
                    os.mkdir(opCtDir)

                meta.__setattr__('PatientID', 'secret')
                meta.__setattr__('PatientBirthDate', '19000101')
                meta.__setattr__('PatientName', 'secret')
                meta.__setattr__('PatientAge', '00Y')
                pdm.write_file(os.path.join(opCtDir, fileCtName + '.dcm'), meta, True)

        dirPetPath = os.path.join(inp, patientFileId, ptDir)
        filePetNames = os.listdir(dirPetPath)
        opPetDir = os.path.join(patientDirPath, 'PET')
        if not os.path.exists(opPetDir):
            os.mkdir(opPetDir)
        for filePetName in filePetNames:
            if filePetName.startswith('PT_'):
                petmeta = pdm.read_file(os.path.join(dirPetPath, filePetName))
                pid = petmeta.PatientID
                petmeta.__setattr__('PatientID', 'secret')
                petmeta.__setattr__('PatientBirthDate', '19000101')
                petmeta.__setattr__('PatientName', 'secret')
                petmeta.__setattr__('PatientAge', '00Y')
                pdm.write_file(os.path.join(opPetDir, filePetName + '.dcm'), petmeta, True)

        # tryDoc = glob.glob(os.path.join(inp, patientFileId, '*.doc'))
        # docExist = len(tryDoc) != 0
        # if docExist:
        #     docSrcPath = tryDoc[0]
        #     docDesPath = os.path.join(patientDirPath, str(index) + 'report.doc')
        #     shutil.copy(docSrcPath, docDesPath)

        menu.append((index, pid + '_' + patientFileId))
        index = index + 1

        # writeDic(menu)

def writeDic(menu):
    pc = os.path.join(opRoot, 'log.txt')
    with open(pc, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(menu)

def loadDic(p):
    '''
    查看自定义id与患者id的对应关系
    :param p: csv文件存储位置
    :return: 二维数组,n*2 每行为“自定义id，患者id”
    '''
    te = []
    import csv
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for lin in reader:
            if not len(lin) == 0:
                te.append(lin)
    return te


if __name__ == '__main__':
    # part1:尚未测试，需要注意的是part1 中ct放在4 中，pet放在文件夹5中
    inp = r'F:\registration\3t'
    opRoot = r'F:\registration\3o'
    maskTheInfo(inp, opRoot)
    # part2:
    # inp = 'F:\dataset\淋巴瘤原始更多\淋巴瘤3'
    # opRoot = 'F:\dataset\淋巴瘤原始更多\抹除患者信息\part2'
    # maskTheInfo(inp, opRoot)
