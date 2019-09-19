import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

csv_file_path = 'train.csv'

mobile_price_classification = pd.read_csv(csv_file_path)

print(mobile_price_classification.head())
print(mobile_price_classification.describe())

features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',  'touch_screen',  'wifi']

X = mobile_price_classification[features]
y = mobile_price_classification.price_range

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# GaussianNB 

'''
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_pred, y_test)

print(acc)
'''

# SVC

'''
clf = SVC(gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_pred, y_test)

print(acc)
'''

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_pred, y_test)

print(acc)


leafs = [None, 10, 100, 200, 300, 400, 500, 1000]


def how_many_leafs(max_leaf_nodes, X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_pred, y_test)
    return acc

for leaf in leafs:
    acc = how_many_leafs(leaf, X_train, X_test, y_train, y_test )
    print('{} leafs | accuracy: {}'.format(leaf, acc))





'''
output = pd.DataFrame({'price':y_test,'predicted_price':y_pred })
output.to_csv('predicted_prices.csv')
'''


'''
NOTES

GaussianNB | acc: 0.8045

SVC(gamma='scale') | acc: 0.9515

DecisionTreeClassifier(random_state=0) | acc: 0.8242
    None leafs | accuracy: 0.8242424242424242
    10 leafs | accuracy: 0.7651515151515151
    100 leafs | accuracy: 0.8181818181818182
    300 leafs | accuracy: 0.8166666666666667
    500 leafs | accuracy: 0.8166666666666667
    1000 leafs | accuracy: 0.8166666666666667

'''