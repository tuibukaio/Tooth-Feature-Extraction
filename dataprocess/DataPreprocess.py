import os
import numpy as np
badlist = ['1005', '1021', '107', '113', '123', '128', 
            '133', '134', '138', '149', '157', '19', 
            '193', '195', '196', '20', '201', '204', 
            '225', '231', '245', '246', '247', '25', 
            '251', '257', '259', '26', '261', '262', 
            '266', '285', '291', '299', '300', '328', 
            '331', '334', '335', '338', '339', '34', 
            '346', '354', '367', '375', '384', '39', 
            '393', '40', '400', '406', '409', '411', 
            '417', '426', '447', '45', '453', '462', 
            '545', '559', '594', '596', '617', '621', 
            '625', '628', '634', '637', '645', '647', 
            '658', '663', '671', '672', '684', '692', 
            '7', '705', '71', '713', '725', '729', 
            '730', '733', '745', '75', '753', '754', 
            '755', '769', '77', '775', '778', '783', 
            '801', '81', '823', '825', '826', '837', 
            '85', '852', '867', '877', '889', '893', 
            '913', '925', '93', '939', '943', '945', 
            '946', '964', '968', '984', '985', '993']

'''
牙齿不够12颗的数据剔除，找到不够的下标
共计120套
['1005', '1021', '107', '113', '123', '128', 
'133', '134', '138', '149', '157', '19', 
'193', '195', '196', '20', '201', '204', 
'225', '231', '245', '246', '247', '25', 
'251', '257', '259', '26', '261', '262', 
'266', '285', '291', '299', '300', '328', 
'331', '334', '335', '338', '339', '34', 
'346', '354', '367', '375', '384', '39', 
'393', '40', '400', '406', '409', '411', 
'417', '426', '447', '45', '453', '462', 
'545', '559', '594', '596', '617', '621', 
'625', '628', '634', '637', '645', '647', 
'658', '663', '671', '672', '684', '692', 
'7', '705', '71', '713', '725', '729', 
'730', '733', '745', '75', '753', '754', 
'755', '769', '77', '775', '778', '783', 
'801', '81', '823', '825', '826', '837', 
'85', '852', '867', '877', '889', '893', 
'913', '925', '93', '939', '943', '945', 
'946', '964', '968', '984', '985', '993']
'''
path = "data_1000/crown_1000/"
cnt = 0
tlist = []
for dirs in os.listdir(path):
    idx = dirs.split("_")[1]
    fs = os.listdir(path + dirs)
    lenth = len(fs)
    for f in fs:
        if not f.startswith('m'):
            lenth -= 1
    if lenth < 12:
        cnt += 1
        tlist.append(idx)
print(cnt)
print(tlist)

'''
计算点云点数最大和最小
计算teeth_1000中牙齿点云文件中最多点云个数为78167，最少为12639，牙齿套数为847(去除120套坏牙)
'''
path = "data_1000/teeth_1000/"
cnt = 0
maxl, minl = 0 , 0 
for line in os.listdir(path):
    if line.startswith('t'):
        continue
    lines = 0
    fname = os.path.splitext(line)[0]
    if fname.split('_')[1] in badlist:
        continue
    for index, line in enumerate(open(path + line,'r')):
        lines += 1
    cnt = cnt + 1
    if cnt == 1:
        maxl = lines
        minl = lines
        continue
    if maxl < lines:
        maxl = lines
    if minl > lines:
        minl = lines
print(cnt)
print(maxl)
print(minl)


'''
去除离群点,暂且去掉heatmap中最大的后200个点
'''

path = "data_1000/teeth_1000/"
root = "data_1000/teeth_1000_without_outlier/"
tot = len(os.listdir(path)) - 120
cnt = 0
for line in os.listdir(path):
    fname = os.path.splitext(line)[0]
    if fname.split('_')[1] in badlist:
        continue
    cnt += 1
    print(str(cnt) + ' / ' + str(tot) + '...')
    point_set = np.loadtxt(path + line)
    idx = np.argsort(point_set[:,3])
    a = point_set[idx]
    idx = idx[-200:]
    point_set = np.delete(point_set, idx, axis = 0)
    np.savetxt(root + line , point_set)
    

'''
对点云文件进行FPS,并保存到teeth_data文件夹下
'''
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

path = "data_1000/teeth_1000_without_outlier/"
root = "data_1000/teeth_data/"
npoints = 1024 * 8
tot = len(os.listdir(path))
cnt = 0
for line in os.listdir(path):
    fname = os.path.splitext(line)[0]
    if fname.split('_')[1] in badlist:
        continue
    cnt += 1
    print(str(cnt) + ' / ' + str(tot) + '...')
    point_set = np.loadtxt(path + line)
    point_set = farthest_point_sample(point_set, npoints)
    np.savetxt(root + line , point_set)
    

'''
将训练和测试用的数据文件进行划分
'''
fr = open("data_1000/teeth_data/teeth_train.txt", "wt")
ft = open("data_1000/teeth_data/teeth_test.txt", "wt")
path = "data_1000/teeth_data/"
for line in os.listdir(path):
    if line.endswith("t.txt") or line.endswith("n.txt"):
        continue
    filename = os.path.splitext(line)[0]
    filenum = filename.split('_')[1]
    if int(filenum) <= 800 and filenum not in badlist:   
        fr.write(filename + "\n")
    else:
        ft.write(filename + "\n")
fr.close()
ft.close()