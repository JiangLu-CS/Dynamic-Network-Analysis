from mainFunc import *
from function import *
from visualizeFunction import *

def main():
    # 读取csv时间序列文件
    asd = batchReadCSVAndParseToArray('D:\github-project\DynamicBrainNetwork-Jianglu\data\group1(isAsd)')
    notAsd = batchReadCSVAndParseToArray('D:\github-project\DynamicBrainNetwork-Jianglu\data\group2(isntAsd)')

    # 数据处理，时间序列转化为动态网络
    asdBrainArray = []
    notAsdBrainArray = []
    for i in range(0, len(asd)):
        asdBrainArray.append(slidingwindow(asd[i], 40, 25, 1))
    for i in range(0, len(notAsd)):
        notAsdBrainArray.append(slidingwindow(notAsd[i], 40, 25, 1))

    #二值化
    asdBiBrainArray = []
    notAsdBiBrainArray = []
    for i in range(0, len(asdBrainArray)):
        asdBiBrainArray.append(toBinary(asdBrainArray[i], 0.7))
    for i in range(0, len(notAsdBrainArray)):
        notAsdBiBrainArray.append(toBinary(notAsdBrainArray[i], 0.7))

    timeDAnalysis(asdBiBrainArray, notAsdBiBrainArray, 'transitivity')
    timeDAnalysis(asdBiBrainArray, notAsdBiBrainArray, 'density')

    #获取contactSequence
    asdContactSequence = []
    notAsdContactSequence = []
    for i in range(0, len(asdBiBrainArray)):
        asdContactSequence.append(GetContactSequence(asdBiBrainArray[i]))
    for i in range(0, len(notAsdBiBrainArray)):
        notAsdContactSequence.append(GetContactSequence(notAsdBiBrainArray[i]))

    ##获取时间聚合网络
    asdAggre = []
    notAsdAggre = []
    for i in range(0, len(asdContactSequence)):
        asdAggre.append(timeAggregatenetwork(asdContactSequence[i], asdBrainArray[i]))
    for i in range(0, len(notAsdContactSequence)):
        notAsdAggre.append(timeAggregatenetwork(notAsdContactSequence[i], notAsdBrainArray[i]))


    #获取时间聚合网络的特征 1.degree t检验
    timeAggregatedAnalysis(asdAggre, notAsdAggre)

if __name__ == '__main__':
    main()

