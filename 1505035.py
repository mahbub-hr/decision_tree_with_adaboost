#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math

#%%
class DecissionTree:
    def __init__(self, depth, target_column, attributes):
        self.max_depth = depth
        self.tree = {}
        self.label = target_column
        self.attributes = attributes

    def entropy(self, values):
        classes , class_count = np.unique(values, return_counts=True)
        total_element = len(values)
        entropy = 0.0
        for i in range(len(classes)):
            p = class_count[i]/total_element
            entropy -= p*np.log2(p)
        return entropy


    def information_gain(self, dataset, feature):
        
        entropy_before = self.entropy(dataset[self.label])
        dfs = dict(tuple(dataset.groupby(feature)))
        feture_length = len(dataset[feature])
        weighted_entropy = 0.0
        for key in dfs:
            df = dfs[key]
            p = df.shape[0]/feture_length  
            weighted_entropy += p*self.entropy(df[self.label])

        return entropy_before - weighted_entropy

    def find_best_attribute(self, examples, attributes):
        size = len(attributes)
        max_gain = None
        best_attribute = None
        for i in range(size):
            info_gain = self.information_gain(examples, attributes[i])
            if max_gain is None or info_gain > max_gain:
                max_gain = info_gain
                best_attribute = attributes[i]
        return best_attribute

    def fit(self, examples, attributes, parent_examples, depth):
        self.tree = self.build_tree(examples, attributes, parent_examples, depth)
        return 

    def build_tree(self, examples, attributes, parent_examples, depth):
        node = {}
        if depth == self.max_depth:
            node['leaf'] = True
            node['class'] = examples[self.label].mode().iloc[0]
            return node

        elif examples.shape[0] == 0:
            node['leaf'] = True
            node['class'] = parent_examples[self.label].mode().iloc[0]
            return node

        elif len(examples.groupby([self.label])) == 1:
            node['leaf'] = True
            node['class'] = examples[self.label].values[0]
            return node

        elif len(attributes) == 0:
            node['leaf'] = True
            node['class'] = examples[self.label].mode().iloc[0]
            return node

        else:
            node['leaf'] = False
            best_attribute = self.find_best_attribute(examples, attributes)
            attributes.remove(best_attribute)
            node['attribute'] = best_attribute
            if best_attribute == 'Dependents':
                print(examples[best_attribute])
            dfs = dict(tuple(examples.groupby(best_attribute)))
            for value in dfs: 
                # print(best_attribute, value)
                subtree = self.build_tree(dfs[value], attributes, examples, depth+1 )
                node[value] = subtree

            return node

        return node

    def single_value_predict(self, instance):
        temp = self.tree
        while not temp['leaf']:
            attribute = temp['attribute']
            # print(instance)
            try:
                temp = temp[instance[attribute].iloc[0]]
            except Exception as e:
                # print(instance)
                # print(attribute, instance[attribute].iloc[0])
                # print(temp)
                # exit()
                pass

        predicted_class = temp['class']
        return predicted_class

    def predict(self, dataframe):
        predicted_classes = []
        for i in range(dataframe.shape[0]):
            predicted_classes.append(self.single_value_predict(dataframe.iloc[[i]]))
        
        return predicted_classes
    def perf_measure(self, y_actual, y_hat, file):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1
        print ('True Positive:',TP)
        print ('False Positive:',FP)
        print ('True Negative:',TN)
        print ('False Negative:',FN)
        accuracy = ((TP+TN)*1.0)/(TP+FP+TN+FN)
        true_positive_rate = (TP*1.0)/(TP+FN)
        true_negative_rate = (TN*1.0)/(TN+FP)
        positive_predictive_value = (TP*1.0)/(TP+FP)
        false_discovery_rate = 1 - positive_predictive_value
        f1_score = 2.0 * ( (TP*2.0) / (TP*2.0 + FP + FN) )
        file.write(f"Accuracy : {round(accuracy,2)}\n")
        file.write(f"True positive rate: {round(true_positive_rate,2)}\n")
        file.write(f"True negative rate: {round(true_negative_rate,2)}\n")
        file.write(f"Positive predictive value: {round(positive_predictive_value,2)}\n")
        file.write(f"False discovery rate: {round(false_discovery_rate,2)}\n")
        file.write(f"F1 Score: {round(f1_score,2)}\n")
        return(TP, FP, TN, FN)

#%%
def adaboost(examples,K,attributes,label):

    w = [1.0/(1.0*examples.shape[0])]*examples.shape[0]
    z = []
    h = []
    discarded = 0
    k = 1
    while k <= K:
        dataframe = examples.copy()
        sampled_frame = dataframe.sample(frac=1,weights=w,replace=True)
        dt = DecissionTree(1, label, attributes)
        dt.fit(sampled_frame, attributes, None, 0)
        # root = buildDecisionTree(sampled_frame,sampled_frame,attributes[:],0,1,label,examples)
        error = 0.0
        pred = dt.predict(dataframe)
        true = dataframe[label].values
        for i in range(dataframe.shape[0]):
            if not true[i]==pred[i]:
                error += w[i]
        if error>0.5:
            discarded += 1
            print('discarded')
            continue
        h.append(dt)
        print(k+1)
        discarded = 0
        for i in range(dataframe.shape[0]):
            if true[i] == pred[i] and not error==0:
                w[i] = w[i]*(error/(1.0-error))
        w = [float(i) / sum(w) for i in w]
        if not error==0:
            weight = (1.0-error)/error
        else:
            weight = float("inf")
        z.append(np.log2(weight))
        k +=1

    return h,z

def replaceZeros(to_replace, value):
    to_replace = list(map(int, to_replace))
    for i in range(len(to_replace)):
        if to_replace[i] == 0:
            to_replace[i] = -1
    return to_replace

def weighted_majority(h,z, examples, label):
    y_actual = examples[label].values
    y_actual = replaceZeros(y_actual,-1)

    all_predicted_values = []
    for i in range(len(h)):
        y_hat = h[i].predict(examples)
        y_hat = replaceZeros(y_hat,-1)
        all_predicted_values.append(y_hat)

    y_hat_final = []

    for i in range(examples.shape[0]):
        weight = 0
        for k in range(len(z)):
            weight += all_predicted_values[k][i]*z[k]
        if weight>=0:
            y_hat_final.append(1)
        else:
            y_hat_final.append(-1)
    count = 0
    for i in range(examples.shape[0]):
        if y_hat_final[i] == y_actual[i]:
            count += 1

    accuracy = (count*100.0)/examples.shape[0]
    print(f'Accuracy:{round(accuracy,2)}%')
    return


#%%
def readTelcoData(n_bin=100):
    telco = pd.read_csv("telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv", delimiter=",");
    non_categorical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    telco.drop('customerID', axis=1, inplace=True)
    print(telco.shape)
    oe = preprocessing.OrdinalEncoder()
    for column in telco:
        if telco[column].isnull().sum() or telco[column].eq(' ').sum():
            print(column, " has missing value:", telco[column].eq(' ').sum())
            if column in non_categorical:
                telco[column] = pd.to_numeric(telco[column], errors='coerce')

    telco.fillna(telco.select_dtypes(include='number').mean().iloc[0], inplace=True)
    telco.fillna(telco.select_dtypes(include='object').mode().iloc[0], inplace=True)

    for column in telco:
        if column in non_categorical:
            telco[column]=pd.cut(telco[column],n_bin, labels=False)
        else: 
            # print(telco[column].dtype)
            telco[column] = oe.fit_transform(telco[[column]])
    telco.reset_index(drop=True, inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(telco['Churn'])
    telco.Churn = le.transform(telco.Churn)
    attributes = list(telco)
    attributes.remove("Churn")

    return telco, 'Churn', attributes

#%%
def readAdultData(file, n_bin=100):
    attributes = ['age','workclass','fnlwgt','education','education-num',
                  'marital-status','occupation','relationship','race','sex',
                  'capital-gain','capital-loss','hours-per-week','native-country','salary']
    non_categorical =['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    label = "salary"
    dataframe = pd.read_csv(file, delimiter=", ",names=attributes,engine='python')
    # print(dataframe)
    print(dataframe.shape)
    dataframe = dataframe.replace('?', pd.np.nan)
    dataframe[label]= dataframe[label].replace('>50K.', 1)
    dataframe[label]= dataframe[label].replace('<=50K.', 0)
    dataframe = dataframe.dropna(axis=0)
    dataframe.reset_index(drop=True, inplace=True)
    oe = preprocessing.OrdinalEncoder()
    for column in dataframe:
        if column in non_categorical:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
            dataframe[column]=pd.cut(dataframe[column],n_bin, labels=False)
        else:
            dataframe[column] = oe.fit_transform(dataframe[[column]])

    print(dataframe.shape)
    attributes.remove(label)
    return dataframe, label, attributes

#%%
def readCreditData(n_bin=100):

    return

#%%
max_depth = 10
train, label, attributes = readAdultData("./adult_dataset/adult.csv",200)
test, label, attributes = readAdultData("./adult_dataset/adult_test.csv",200)
# print(dataset.Churn)
# train, test = train_test_split(dataset, test_size=0.2)
dt = DecissionTree(max_depth, label, attributes)
dt.fit(train, attributes, None, 0)
print("====== training finished =====")
with open("performance.txt",'a') as file:
    # file.write("Train\n=============================>\n")
    # y_hat = dt.predict(train)
    # dt.perf_measure(train[label].values, y_hat, file)
    file.write("Test\n=============================>\n")
    y_hat = dt.predict(test)
    dt.perf_measure(test[label].values, y_hat, file)

print("<==============Finished ================>")

#%%
#h,z=adaboost(train, 5, attributes, label)
# %%
train, label, attributes = readAdultData("./adult_dataset/adult.csv",200)
test, label, attributes = readAdultData("./adult_dataset/adult_test.csv",200)
print(train.shape)
k=5
print("<==============Start ================>")
for k in range(k, k+1):
    h, z = adaboost(train, k, attributes, label)
    print(f"<==============Adaboost {k} Finished ================>")
    weighted_majority(h,z, test,label)

    print(f"<==============Majority Voting for {k} Finished ================>")
    k+=5
# %%

# %%
