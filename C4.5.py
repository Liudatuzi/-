from math import  log
from operator import  *
from sklearn.model_selection import train_test_split
def calEntropy(dataSet):
    #数据集行数
    numEntry=len(dataSet)
    labelCount={}
    for vector in dataSet:
        currentLabel=vector[-1]#某一行数据的最后一个标签
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    Entropy=0.0
    for key in labelCount:
        prob=float(labelCount[key])/numEntry
        Entropy-=prob*log(prob,2)
    return Entropy


def load_file(file_name):
    f = open(file_name)  # 打开训练数据集所在的文档
    feature = []  # 存放特征的列表

    for row in f.readlines():
        number = row.strip().split(",")  # 按照\t分割每行的元素，得到每行特征和标签
        feature.append(number)
    f.close()  # 关闭文件，很重要的操作
    return feature

def createDataset():
    str = "breast-cancer.data"
    dataSet = load_file(str)
    label = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad",
             "irradiat"]
    return dataSet, label
def splitDataSet(dataSet,axis,value):
    #分割数据集
    retDataset=[]
    for vector in dataSet:
        if vector[axis]==value:
            newVector=vector[:axis]
            #去掉axis特征
            newVector.extend(vector[axis+1:])
            retDataset.append(newVector)
    return retDataset

def chooseBestFeature(dataSet):
    numOfFeature=len(dataSet[0])-1#without consideration to the last axis
    baseEntropy=calEntropy(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numOfFeature):
        feaList=[example[i] for example in dataSet]
        unique=set(feaList)
        #经验条件熵
        newEntropy=0.0
        spilit_info=0.0
        for value in unique:
            subDataSet=splitDataSet(dataSet,i,value)
            sub=len(subDataSet)
            total=len(dataSet)
            prob=float(sub)/total
            newEntropy+=prob*calEntropy(subDataSet)
            spilit_info+=-prob*log(prob,2)
        #信息增益,当差值越大说明按照此划分对于事件的混乱程度减少越有帮助。
        info_gain=baseEntropy-newEntropy
        if spilit_info==0:
            continue
        #信息增益比
        info_gain_ratio=info_gain/spilit_info
        print(i," ",info_gain)
        if info_gain_ratio>bestInfoGain:
            bestInfoGain=info_gain_ratio
            bestFeature=i
    return bestFeature
#calculate the max times class
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        #降序排列
        sortedClassCount=sorted(classCount.items() , key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #if the example all the same
    if classList.count(classList[-1])==len(classList):
        return classList[-1]#all the class are the same
    #not sure
    #if len(classList[0]) == 1:
       # return majorityCnt(classList)

    #choose the largest entropy
    bestFeature=chooseBestFeature(dataSet)
    bestLabel=labels[bestFeature]
    myTree={bestLabel:{}}
    del(labels[bestFeature])
    featureValue=[example[bestFeature] for example in dataSet]
    uniqueValue=set(featureValue)
    for value in uniqueValue:
        subLabels=labels[:]
        newSet=splitDataSet(dataSet,bestFeature,value)
        myTree[bestLabel][value]=createTree(newSet,subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    predict_class = -1
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                predict_class = classify(secondDict[key], featLabels, testVec)
            else: predict_class = secondDict[key]
    return predict_class


if __name__=="__main__":
    dataSet, labels = createDataset()

    X_train, X_test = train_test_split(dataSet, test_size=0.4, random_state=0)

    label = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
             'irradiat']
    tree = createTree(X_train, labels)

    print(['no-recurrence-events', '30-39', 'premeno', '30-34', '0-2', 'no', '3', 'left', 'left_low', 'no'])
    print(X_train[0])

    accuracy = 0.0
    total = len(X_test)
    correct = 0

    for vector in X_test:
        b = classify(tree, label, vector)
        if b == vector[-1]:
            correct += 1
    accuracy = float(correct) / total
    print(accuracy)
    # print(tree)



