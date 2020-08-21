import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Output indexes
lid = test.L_Id

# Removing Columns
def remove_unrelated(df):
    df.drop(["Date of Creation","Estimated Date of Completion","Actual Date of Completion","Agent Category Assigned","L_Id"],axis = 1, inplace = True)
    

# Clean train and test
remove_unrelated(train)
train.dropna(axis = 0,inplace = True)
remove_unrelated(test)

# Deciding X and Y
X = train.drop("Problem Category", axis = 1)
Y = train['Problem Category']

# Labelling continous and categorical data
cat = ["Type of Request","Description of the Request","Location",'Street Type',"Region Type","Ward No","Request Solution Category","Team Assigned"]
con = ["A_1","A_2"]

# Creating ML pipeline
num_transformer = Pipeline(steps=[
    ('imputer',KNNImputer(n_neighbors = 5)),('scaler',StandardScaler())
])

cat_transformer = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy = 'most_frequent',fill_value = 'missing')),('onehot',OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num',num_transformer,con),('cat',cat_transformer,cat)
    ]
)

estimator = [
    LogisticRegression(penalty = 'l2', solver = 'newton-cg'),
    DecisionTreeClassifier(criterion = "entropy",splitter = "best"),
    RandomForestClassifier(n_estimators = 10)
]

model_pipe = Pipeline(steps = [
    ('preprocess', preprocessor),
    ('Boosting', AdaBoostClassifier(base_estimator = estimator[1] ,n_estimators = 170))])

# Fitting and getting output
model_pipe.fit(X,Y)
output = model_pipe.predict(test)

# convert this array to data frame and then write it in file
out = pd.DataFrame({"L_Id":lid[:],"Problem Category":output[:]})
#out.set_index("L_Id")

#If 'df' is the dataframe containing your predictions and no feature column is for index:
out.to_csv('submission.csv', index= False)
