from dyconnmap import sliding_window
from dyconnmap.fc import PLV

#输入：时间序列
def slidingwindow(fmri, n, window_length, step):
    estimator = PLV()
    toTest = fmri[:, 0:n]
    dfcg = sliding_window(toTest, estimator, window_length=window_length, step=step, pairs=None)
    dfcg_r = np.real(dfcg)
    dfcg_r = np.float32(dfcg_r)
    dfcg_r = np.nan_to_num(dfcg_r)

    for X in range(len(dfcg_r)):
        dfcg_r[X] += dfcg_r[X].T - np.diag(dfcg_r[X].diagonal())
    return dfcg_r

#网络二值化
def toBinary(network, n):
    network[network > n] = 1
    network[network <= n] = 0
    return network

# 获取交流顺序结构 输入t：动态网络
def GetContactSequence(t):
    ContactSequence = []
    for time in range(len(t)):
        for i in range(len(t[0])):
            for j in range(i+1, len(t[0])):
                if t[time][i][j] == 1:
                    tmplist = []
                    tmplist.append(i)
                    tmplist.append(j)
                    tmplist.append(time)
                    ContactSequence.append(tmplist)
    return ContactSequence

#获取强连接节点对 输入交流顺序结构，二维数组
def GetStronglyConnectedNode(ContactSequence):
    StronglyConnectedNode = []
    for i in range(len(ContactSequence)):
        tmplist = set()
        tmplist.add(ContactSequence[i][0])
        tmplist.add(ContactSequence[i][1])
        if tmplist not in StronglyConnectedNode:
            StronglyConnectedNode.append(tmplist)
    return StronglyConnectedNode

# 弱连接节点对 无直接连接，但是有传递性，且限制于时间先后 输入交流顺序结构，二维数组
def GetWeaklyConnectedNode(ContactSequence):
    WeaklyConnectedNode = []
    for i in range(len(ContactSequence)):
        for j in range(i, len(ContactSequence)):
            if ContactSequence[i][1] == ContactSequence[j][0]:
                tmplist = []
                tmplist.append(ContactSequence[i][0])
                tmplist.append(ContactSequence[j][1])
                if tmplist not in WeaklyConnectedNode:
                    WeaklyConnectedNode.append(tmplist)
    return WeaklyConnectedNode


# 某个节点的源集向量 即从节点 i 开始的节点集可以通过在时间 t 或之后开始的时间尊重路径达到
def SourceSetOfSomeNode(n,ContactSequence):
    SourceSet = set()
    StronglyConnectedNode = GetStronglyConnectedNode(ContactSequence)
    WeaklyConnectedNode = GetWeaklyConnectedNode(ContactSequence)
    for i in range(len(StronglyConnectedNode)):
        if n in StronglyConnectedNode[i]:
            for source in StronglyConnectedNode[i]:
                SourceSet.add(source)
    for i in range(len(WeaklyConnectedNode)):
        tmplist = []
        tmplist.append(WeaklyConnectedNode[i][0])
        tmplist.append(n)
        if tmplist in WeaklyConnectedNode:
            SourceSet.add(WeaklyConnectedNode[i][0])

    return SourceSet

# 某个节点的影响向量 即从节点 i 开始的节点集可以通过在时间 t 或之后开始的时间尊重路径达到
def SetOfInfluence(n, ContactSequence):
    SetOfInfluence = set()
    StronglyConnectedNode = GetStronglyConnectedNode(ContactSequence)
    WeaklyConnectedNode = GetWeaklyConnectedNode(ContactSequence)
    for i in range(len(StronglyConnectedNode)):
        if n in StronglyConnectedNode[i]:
            for source in StronglyConnectedNode[i]:
                SetOfInfluence.add(source)
    for i in range(len(WeaklyConnectedNode)):
        tmplist = []
        tmplist.append(n)
        tmplist.append(WeaklyConnectedNode[i][1])
        if tmplist in WeaklyConnectedNode:
            SetOfInfluence.add(WeaklyConnectedNode[i][1])
    return SetOfInfluence

#输入：节点i，节点j，contactSequence。输出：从i到j的最少t（时间尊重路径）
def forwardLantancy(i,j, ContactSequence):
    setExsist = SetOfInfluence(i)
    if i not in setExsist:
        return -2 # 不可达
    setCurrent = set()
    setCurrent.add(i)
    t = 0
    for a in range(len(ContactSequence)):
        if i == ContactSequence[a][0] or i == ContactSequence[a][1]:
            t = ContactSequence[a][2]
            break;
    for a in range(len(ContactSequence)):
        for b in setCurrent:
            if b == ContactSequence[a][0]:
                if j == ContactSequence[a][1]:
                    return ContactSequence[a][2] - t + 1
            if b == ContactSequence[a][1]:
                if j == ContactSequence[a][0]:
                    return ContactSequence[a][2] - t + 1
        if ContactSequence[a][0] in setCurrent:
            setCurrent.add(ContactSequence[a][1])
    return -1

#接近中心性
def closenessCentrality(i,Nnodes):
    latencySum = 0
    for a in range(Nnodes):
        if a != i:
            if forwardLantancy(i,a) != -1 and forwardLantancy(i,a) != -2:
                latencySum += forwardLantancy(i,a)
    return (Nnodes - 1) * (1 / latencySum)

# times 重新布线次数
import random
def randonmizedEdges(contactSequence, times):
    randomcontactSequence = contactSequence
    for a in range(0, times):
        tmpCon = randomcontactSequence
        tmpRanCon = []
        for b in range(len(tmpCon)):
            tmp = []
            i = random.randint(0, 1)
            if i == 0:
                toVerse = random.randint(0, 9)
                while toVerse == tmpCon[b][1]:
                    toVerse = random.randint(0, 9)
                tmp.append(tmpCon[b][0])
                tmp.append(toVerse)
                tmp.append(tmpCon[b][2])
            else:
                toVerse = random.randint(0, 9)
                while toVerse == tmpCon[b][0]:
                    toVerse = random.randint(0, 9)
                tmp.append(toVerse)
                tmp.append(tmpCon[b][1])
                tmp.append(tmpCon[b][2])
            tmpRanCon.append(tmp)
            randomcontactSequence = tmpRanCon
    return randomcontactSequence


def takeSecond(elem):
    return elem[2]

#获取时间随机性的动态网络
def randonmizedPermutedTime(ContactSequence, time):
    randomcontactSequence = []
    for b in range(len(ContactSequence)):
        tmp = []
        tmp.append(ContactSequence[b][0])
        tmp.append(ContactSequence[b][1])
        tmp.append(random.randint(0, time - 1))
        randomcontactSequence.append(tmp)
    randomcontactSequence.sort(key= takeSecond,reverse=False)
    return randomcontactSequence

import numpy as np
#contactsequence转网络,时间限度
def transferContactSequence(contactSequence, time):
    t = np.zeros((len(contactSequence),len(contactSequence[0]),len(contactSequence[0])))
    for i in range(len(contactSequence)):
        t[contactSequence[i][2]][contactSequence[i][0]][contactSequence[i][1]] = t[contactSequence[i][2]][contactSequence[i][1]][contactSequence[i][0]] = 1
    return t

# 可视化动态网络
import matplotlib.pyplot as plt
import networkx as nx
#时间聚集网络可视化
def timeAggregatenetwork(contactSequence, g):
    a = len(g[0])
    t = np.zeros((a,a))
    for i in range(len(contactSequence)):
        t[contactSequence[i][0]][contactSequence[i][1]] = 1
        t[contactSequence[i][1]][contactSequence[i][0]] = 1
    return t