# 可视化动态网络
import matplotlib.pyplot as plt
import networkx as nx

# 可视化动态网络。 输入 t:动态网络，三维矩阵；s：title
def visualizeDynamicNetwork(t, s):
    plt.subplots(4, 5, figsize=(150, 150))
    for ind in range(len(t)):
        plt.subplot(4, 5, ind + 1)
        graph = nx.from_numpy_matrix(t[ind])
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        plt.title('t = %s' % ind)

    plt.suptitle(s, fontsize=30)
    plt.show()

#可视化时间聚合网络
def visualizeSingleNetwork(t):
    graph = nx.from_numpy_matrix(t)
    pos = nx.circular_layout(graph)
    nx.draw(graph,pos, with_labels=True)
    plt.title('aggregrate')
    plt.show()

#输入fmri 时间序列，n 可视化的序列数
def visualizeSeries(fmri, n):
    toTest = fmri[:, 0:n]
    plt.plot(toTest.T)
    plt.show()

#输入某一时刻网络 输出可视化矩阵
def visualizeMatrix(matrix):
    plt.matshow(matrix)

#某一时刻网络的社区性
from networkx.algorithms.community import k_clique_communities
def visualizeCommunity(g, edgeNum):
    plt.figure(figsize=(5,5))
    G = nx.from_numpy_matrix(g)
    klist = list(k_clique_communities(G,edgeNum))
    pos = nx.circular_layout(G)
    COLORS = ['blue', 'yellow', 'green', 'red', 'purple', 'pink', 'black']
    plt.clf()
    nx.draw(G,pos = pos, with_labels=True)
    for i in range(len(klist)):
        nx.draw(G, pos = pos, nodelist = klist[i], node_color = COLORS[i])
    plt.show()

#某一时刻脑图visualize,msdl
from nilearn import plotting
from nilearn import datasets
def visualizeBrainNetwork(brainNetwork):
    atlas_data = datasets.fetch_atlas_msdl()
    atlas_filename = atlas_data.maps
    coords = atlas_data.region_coords
    plotting.plot_connectome(brainNetwork, coords, edge_threshold="50%")
    plotting.show()

import matplotlib.pyplot as plt
def visualizeFeatureComparison(group1, group2, string):
    x = []
    for i in range(0, len(group1)):
        x.append(i)
    plt.title('脑区值对比图:' + string)  # 折线图标题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plt.xlabel('受试者')  # x轴标题
    plt.ylabel('值')  # y轴标题
    plt.plot(x, group1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, group2, marker='o', markersize=3)

    for a, b in zip(x, group1):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    for a, b in zip(x, group2):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.legend(['is Asd', 'isnt Asd'])  # 设置折线名称

    plt.show()  # 显示折线图