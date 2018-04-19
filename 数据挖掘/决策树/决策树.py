import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import tree
import graphviz



test=pd.DataFrame(pd.read_csv('TEST_ML_v2.csv',header=0,encoding='GBK'))
X_df=test[['City', 'Item category', 'Period', 'Gender', 'Age', 'Market channels', 'Self-agent', 'Category', 'Loan channels']]
X_list=X_df.to_dict(orient="records")
vec = DictVectorizer()
X=vec.fit_transform(X_list)
Y=np.array(test['Status'])
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X.toarray(),Y,test_size=0.4,random_state=0)
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

clf.score(X_test, y_test)
0.85444078947368418
clf.predict(X_test[0]),y_test[0](array(['Charged Off'], dtype=object), 'Charged Off')
clf.predict_proba(X_test[0])array([[ 1., 0.]])
clf.classes_array(['Charged Off', 'Fully Paid'], dtype=object)
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=vec.get_feature_names(),
class_names=clf.classes_,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
graph.render("test_e1")