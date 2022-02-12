#流程：1.输入多组样本的脑序列 2.滑动窗口建模01脑网络 3.分析脑网络特征 4.t检验获取显著特征
#分析维度 脑区维度（每个脑区一个值）/聚合维度（一整个动态网络一个值）/ 时间维度（一个特定时间网络一个值）
#时间聚合的分析

import pandas as pd
import numpy as np
import glob, os
from visualizeFunction import *
from nilearn import datasets

def getBrainRegion(i):
    atlas_data = datasets.fetch_atlas_msdl()
    atlas_filename = atlas_data.maps
    labels = atlas_data.labels
    return labels[i]

def batchReadCSVAndParseToArray(path):
    file = glob.glob(os.path.join(path, "*.csv"))
    #print(file)
    dl = []
    for f in file:
        dl.append(pd.read_csv(f))
    fmri = []
    for d in dl:
        fmri.append(np.array(d).T)
    return fmri

import networkx as nx

def groupSingleNetworkFeature(group1, group2, feature):
    res = []
    asdNetworkFeature = []
    notAsdNetworkFeature = []
    if feature == 'transitivity':
        for i in range(len(group1)):
            graph1 = nx.from_numpy_matrix(group1[i])
            asdNetworkFeature.append(nx.transitivity(graph1))
        for i in range(len(group2)):
            graph2 = nx.from_numpy_matrix(group2[i])
            notAsdNetworkFeature.append(nx.transitivity(graph2))
        res.append(asdNetworkFeature)
        res.append(notAsdNetworkFeature)
        return res
    if feature == 'density':
        for i in range(len(group1)):
            graph1 = nx.from_numpy_matrix(group1[i])
            asdNetworkFeature.append(nx.density(graph1))
        for i in range(len(group2)):
            graph2 = nx.from_numpy_matrix(group2[i])
            notAsdNetworkFeature.append(nx.density(graph2))
        res.append(asdNetworkFeature)
        res.append(notAsdNetworkFeature)
        return res

def compareSingleNetwork(group1, group2, n, string):
    if (len(group1) != len(group2)):
        print('len not equal')
        return
    (statistic, pvalue) = scipy.stats.ttest_ind(group1, group2)

    print(string + '有显著特征, pvalue：' + str(pvalue) + ', 组数据对比：')
    print(group1)
    print(group2)

    if pvalue < n:
        print(string + '有显著特征, pvalue：' + str(pvalue) + ', 组数据对比：')
        print(group1)
        print(group2)
        visualizeFeatureComparison(group1, group2, string)

def groupPointFeature(group1, group2, feature):
    res = []
    # 节点个数
    groupNum = len(group1)
    nNode = len(group1[0])
    # 每个节点都有一个数组[(a,b,c),(a,b,c)]
    asdNodeFeature = []
    notAsdNodeFeature = []

    if feature == 'betweeness_centrality':
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.betweenness_centrality(graph1)[i])
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.betweenness_centrality(graph2)[i])
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)
        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        return res



    if feature == 'eigenvector_centrality_numpy':
        degreegroup1 = []
        degreegroup2 = []
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.eigenvector_centrality_numpy(graph1)[i])
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.eigenvector_centrality_numpy(graph2)[i])
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)
        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        return res

    if feature == 'eigenvector_centrality':
        degreegroup1 = []
        degreegroup2 = []
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.eigenvector_centrality(graph1)[i])
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.eigenvector_centrality(graph2)[i])
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)
        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        return res

    if feature == 'degree':
        degreegroup1 = []
        degreegroup2 = []
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.degree(graph1, i))
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.degree(graph2, i))
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)
        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        return res
    if feature == 'closeness_centrality':
        degreegroup1 = []
        degreegroup2 = []
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.closeness_centrality(graph1, i))
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.closeness_centrality(graph2, i))
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)

        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        print(asdNodeFeature)
        print(notAsdNodeFeature)
        return res
    if feature == 'clustering':
        degreegroup1 = []
        degreegroup2 = []
        for i in range(nNode):
            tempNode1 = []
            tempNode2 = []
            for j in range(groupNum):
                graph1 = nx.from_numpy_matrix(group1[j])
                tempNode1.append(nx.clustering(graph1, i))
                graph2 = nx.from_numpy_matrix(group2[j])
                tempNode2.append(nx.clustering(graph2, i))
            asdNodeFeature.append(tempNode1)
            notAsdNodeFeature.append(tempNode2)

        res.append(asdNodeFeature)
        res.append(notAsdNodeFeature)
        print(asdNodeFeature)
        print(notAsdNodeFeature)
        return res
    return

import scipy
#比较脑区之间的特征
def compareRegion(group1, group2, n, string):
    if(len(group1) != len(group2)):
        print('len not equal')
        return
    for i in range(len(group1)):
        (statistic, pvalue) = scipy.stats.ttest_ind(group1[i], group2[i])
        if pvalue < n:
            print(string +  '在' + getBrainRegion(i) + '脑区有显著特征, pvalue：' + str(pvalue) + ', 组数据对比：')
            print(group1[i])
            print(group2[i])
            visualizeFeatureComparison(group1[i], group2[i], string)


#def pointDAnalysis(asdAggre, notAsdAggre):


#以时间为维度的分析
def timeDAnalysis(asdBiBrainArray, notAsdBiBrainArray, n, string):
    # transitivity
    if string == 'transitivity':
        timeDAnalysisHelper(asdBiBrainArray, notAsdBiBrainArray, n, 'transitivity')
    # density
    if string == 'density':
        timeDAnalysisHelper(asdBiBrainArray, notAsdBiBrainArray, n, 'density')


def timeDAnalysisHelper(asdBiBrainArray, notAsdBiBrainArray, n, string):
    asdGroup = []
    notAsdGroup = []
    if string == 'transitivity':
        for i in range(len(asdBiBrainArray)):
            tempNetwork1 = []
            for j in range(len(asdBiBrainArray[i])):
                graph1 = nx.from_numpy_matrix(asdBiBrainArray[i][j])
                tempNetwork1.append(nx.transitivity(graph1))
            asdGroup.append(tempNetwork1)

        for i in range(len(notAsdBiBrainArray)):
            tempNetwork2 = []
            for j in range(len(notAsdBiBrainArray[i])):
                graph2 = nx.from_numpy_matrix(notAsdBiBrainArray[i][j])
                tempNetwork2.append(nx.transitivity(graph2))
            notAsdGroup.append(tempNetwork2)

        toCompare = []
        for i in range(len(asdGroup[0])):
            temp1 = []
            temp2 = []
            temp = []
            for j in range(len(asdGroup)):
                temp1.append(asdGroup[j][i])
                temp2.append(notAsdGroup[j][i])
            temp.append(temp1)
            temp.append(temp2)
            toCompare.append(temp)

        for i in range(len(toCompare)):
            (statistic, pvalue) = scipy.stats.ttest_ind(toCompare[i][0], toCompare[i][1])
            if pvalue < n:
                print( 'transitivity在' + str(i) + '时刻有显著特征, pvalue：' + str(pvalue) + ', 组数据对比：')
                print(toCompare[i][0])
                print(toCompare[i][1])
                visualizeFeatureComparison(toCompare[i][0], toCompare[i][1], 'ccc')

    if string == 'density':
        for i in range(len(asdBiBrainArray)):
            tempNetwork1 = []
            for j in range(len(asdBiBrainArray[i])):
                graph1 = nx.from_numpy_matrix(asdBiBrainArray[i][j])
                tempNetwork1.append(nx.density(graph1))
            asdGroup.append(tempNetwork1)

        for i in range(len(notAsdBiBrainArray)):
            tempNetwork2 = []
            for j in range(len(notAsdBiBrainArray[i])):
                graph2 = nx.from_numpy_matrix(notAsdBiBrainArray[i][j])
                tempNetwork2.append(nx.density(graph2))
            notAsdGroup.append(tempNetwork2)

        toCompare = []
        for i in range(len(asdGroup[0])):
            temp1 = []
            temp2 = []
            temp = []
            for j in range(len(asdGroup)):
                temp1.append(asdGroup[j][i])
                temp2.append(notAsdGroup[j][i])
            temp.append(temp1)
            temp.append(temp2)
            toCompare.append(temp)

        for i in range(len(toCompare)):
            (statistic, pvalue) = scipy.stats.ttest_ind(toCompare[i][0], toCompare[i][1])
            if pvalue < 0.05:
                print( 'density' + str(i) + '时刻有显著特征, pvalue：' + str(pvalue) + ', 组数据对比：')
                print(toCompare[i][0])
                print(toCompare[i][1])
                visualizeFeatureComparison(toCompare[i][0], toCompare[i][1], 'ccc')

#时间聚合网络分析
def timeAggregatedAnalysis(asdAggre, notAsdAggre, string):
    #transitivity
    if string == 'transitivity':
        transitivityFeature = groupSingleNetworkFeature(asdAggre, notAsdAggre, 'transitivity')
        compareSingleNetwork(transitivityFeature[0], transitivityFeature[1], 0.01, 'transitivity')

    #density
    if string == 'density':
        densityFeature = groupSingleNetworkFeature(asdAggre, notAsdAggre, 'density')
        compareSingleNetwork(densityFeature[0], densityFeature[1], 0.05, 'density')

    #degree
    if string == 'degree':
        degreeFeature = groupPointFeature(asdAggre, notAsdAggre, 'degree')
        compareRegion(degreeFeature[0], degreeFeature[1], 0.01, 'degree')

    #closeness_centrality
    if string == 'closeness_centrality':
        closenessFeature = groupPointFeature(asdAggre, notAsdAggre, 'closeness_centrality')
        compareRegion(closenessFeature[0], closenessFeature[1], 0.01, 'closeness_centrality')

    #clustering
    if string == 'clustering':
        clusteringFeature = groupPointFeature(asdAggre, notAsdAggre, 'clustering')
        compareRegion(clusteringFeature[0], clusteringFeature[1], 0.01, 'clustering')

    #eigenvector_centrality_numpy
    if string == 'eigenvector_centrality_numpy':
        eigenvector_centrality_numpyFeature = groupPointFeature(asdAggre, notAsdAggre, 'eigenvector_centrality_numpy')
        compareRegion(eigenvector_centrality_numpyFeature[0], eigenvector_centrality_numpyFeature[1], 0.05, 'eigenvector_centrality_numpy')

    #eigenvector_centrality
    if string == 'eigenvector_centrality':
        eigenvector_centralityFeature = groupPointFeature(asdAggre, notAsdAggre, 'eigenvector_centrality')
        compareRegion(eigenvector_centralityFeature[0], eigenvector_centralityFeature[1], 0.05, 'eigenvector_centrality')

    #betweeness_centrality
    if string == 'betweeness_centrality':
        betweeness_centralityFeature = groupPointFeature(asdAggre, notAsdAggre, 'betweeness_centrality')
        compareRegion(betweeness_centralityFeature[0], betweeness_centralityFeature[1], 0.05, 'betweeness_centrality')
