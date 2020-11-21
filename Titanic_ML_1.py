import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(10)

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head(10)

women = train_data.loc[train_data.Pclass == 3]["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of women who survived:", rate_men)

def preprocess(data, tL, flag = 0):
    X = pd.get_dummies(data[features])
    df = pd.get_dummies(X['Pclass'].astype(str), prefix = 'Pclass')
    X = pd.concat([X, df], axis=1)
    del X['Pclass']

    #df = pd.get_dummies(X['SibSp'].astype(str), prefix = 'SibSp')
    #X = pd.concat([X, df], axis=1)
    X['SibSp_0'] = X['SibSp'] == 0
    X['SibSp_1'] = X['SibSp'] == 1
    X['SibSp_2'] = X['SibSp'] == 2
    X['SibSp_3'] = X['SibSp'] > 2
    del X['SibSp']
    #df = pd.get_dummies(X['Parch'].astype(str), prefix = 'Parch')
    #X = pd.concat([X, df], axis=1)
    X['Parch_0'] = X['Parch'] == 0
    X['Parch_1'] = X['Parch'] == 1
    X['Parch_2'] = X['Parch'] == 2
    X['Parch_3'] = X['Parch'] > 2
    del X['Parch']
    #print(df)

    meanM = X[X.Sex_female == 1]['Age'].mean()
    meanW = X[X.Sex_female == 0]['Age'].mean()

    indices = X['Sex_female'] == 1
    X.loc[indices, 'Age'] = X.loc[indices, 'Age'].fillna(meanW)

    indices = X['Sex_male'] == 1
    X.loc[indices, 'Age'] = X.loc[indices, 'Age'].fillna(meanM)
        X['Age_1_10'] = X['Age'] < 10
    X['Age_10_20'] = (X['Age'] >= 10) & (X['Age'] < 20)
    X['Age_20_30'] = (X['Age'] >= 20) & (X['Age'] < 30)
    X['Age_30_40'] = (X['Age'] >= 30) & (X['Age'] < 40)
    X['Age_41'] = X['Age'] >= 40
    X["Age"] = X["Age"] / X["Age"].max()
    del X['Age']

    X['tid'] = data['Ticket'].fillna("0").str.replace(' ', '', regex=True)
    #print(X['tid'][772])
    X['tid'] = X['tid'].str.replace('S\.O\.P\.', '', regex=True)
    X['tid'] = X['tid'].str.replace('A5', '', regex=True)
    X['tid'] = X['tid'].str.replace('A\.5\.', '', regex=True)
    X['tid'] = X['tid'].str.replace('A\/5\.', '', regex=True)
    X['tid'] = X['tid'].str.replace('CA\.', '', regex=True)
    X['tid'] = X['tid'].str.replace('C\.A\.', '', regex=True)
    X['tid'] = X['tid'].str.replace('O2\.', '', regex=True)

    X['tid'] = X['tid'].str.extract('(\d+)').fillna("1000000").astype(int)
    indices = (X['SibSp_0'] == True) & (X['Parch_0'] == True)
    X.loc[indices, 'tid'] = 0
    if flag == 1:
        indices = data['Survived'] == 0 
        X.loc[indices, 'tid'] = 0
    
    temp = X['tid'].drop_duplicates(keep=False).tolist()
    for index, row in X.iterrows():
        if row['tid'] in temp:
            print(row['tid'])
            X.at[index,'tid'] = 0
            
    lb = preprocessing.LabelBinarizer()
    #X['tid'] = lb.fit_transform(X['tid'])
    df = X['tid']
    if flag == 1:
        
        df = pd.DataFrame(lb.fit_transform(df),
                          columns=lb.classes_, 
                          index=df.index)
        tL = lb.classes_
        X = pd.concat([X, df], axis=1)
    else:
        for ll in tL:
            X[ll] = X['tid'] == ll
    del X['tid']
    #print("---")
    
    mean1 = X[X.Pclass_1 == 1]['Fare'].mean()
    mean2 = X[X.Pclass_2 == 1]['Fare'].mean()
    mean3 = X[X.Pclass_3 == 1]['Fare'].mean()

    indices = X['Pclass_1'] == 1
    X.loc[indices, 'Fare'] = X.loc[indices, 'Fare'].fillna(mean1)
    indices = X['Pclass_2'] == 1
    X.loc[indices, 'Fare'] = X.loc[indices, 'Fare'].fillna(mean2)
    indices = X['Pclass_3'] == 1
    X.loc[indices, 'Fare'] = X.loc[indices, 'Fare'].fillna(mean3)

    X['Fare_1_5'] = X['Fare'] < 5
    X['Fare_6_10'] = (X['Fare'] >= 5) & (X['Fare'] < 10)
    X['Fare_10_15'] = (X['Fare'] >= 10) & (X['Fare'] < 15)
    X['Fare_16_20'] = (X['Fare'] >= 15) & (X['Fare'] < 20)
    X['Fare_21_25'] = (X['Fare'] >= 20) & (X['Fare'] < 25)
    X['Fare_26_30'] = (X['Fare'] >= 25) & (X['Fare'] < 30)
    X['Fare_31_35'] = (X['Fare'] >= 30) & (X['Fare'] < 35)
    X['Fare_36_40'] = (X['Fare'] >= 35) & (X['Fare'] < 40)
    X['Fare_41'] = X['Fare'] >= 40
    X["Fare"] = X["Fare"] / X["Fare"].max()
    del X['Fare']
    
     X['title'] = data['Name'].str.extract(r'\w+, (\w*)')
    #df = X['title']
    #df = pd.DataFrame(lb.fit_transform(df),
    #                      columns=lb.classes_, 
    #                      index=df.index)
    #X = pd.concat([X, df], axis=1)
    #indices = (X['Rev'] == 1) | (X['sir'] == 1) | (X['the'] == 1) 
    X['Mr'] = X['title'] == 'Mr'
    X['Mrs'] = X['title'] == 'Mrs'
    X['Miss'] = X['title'] == 'Miss'
    X['Master'] = X['title'] == 'Master'
    X['Officer'] = X['title'] == 'Officer'
    del X['title']
    return X, tL
    
   from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]
X = ""
tL = []
X, tL = preprocess(train_data, tL, 1)
print(list(X.columns))
print("Preprocessing Done\n")
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=12)

parameter_space = {
    'hidden_layer_sizes': [(1,), (100,20), (70,20),(150,20),(50,20)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'ibfgs'],
    'alpha': [0.1],
    'learning_rate': ['constant','adaptive','invscaling'],
    'learning_rate_init' : [0.08, 0.09, 0.1, 0.11,0.14,0.18,0.2,0.21,0.3,0.35],
}
#model = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(20,15), random_state=1, learning_rate_init=l2/100, max_iter = 500)
model = MLPClassifier(max_iter=1500)
#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=2)
clf.fit(X_train, y_train)
# Best paramete set
print('Best parameters found:\n', clf.best_params_)

y_pred = clf.predict(X_dev)
y_trp = clf.predict(X_train)


print(X_dev)
print(y_dev)
print(y_pred)

accuracy = np.sum(np.abs(y_dev.to_numpy() - y_pred))
accuracyt = np.sum(np.abs(y_train.to_numpy() - y_trp))
#print(y_pred)
#print(y_dev.to_numpy())
#print(accuracy)
print(accuracy / len(y_dev.to_numpy()), accuracyt / len(y_train.to_numpy()))


kf = KFold(n_splits=2,shuffle=False)
kf.split(X)    
     
# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model = []
 
# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the model
    model = clf.fit(X_train, y_train)
    # Append to accuracy_model the accuracy of the model
    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

# Print the accuracy    
print(accuracy_model)

print(tL)
X1, tL2 = preprocess(test_data, tL, 0)
print("----------------")
print(X1)
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]
predictions = clf.predict(X1)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
