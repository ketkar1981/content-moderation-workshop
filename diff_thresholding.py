import pandas as pd
import io
import requests


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split, GroupKFold
from mlxtend.classifier import StackingCVClassifier
from sklearn.feature_selection import SelectFromModel

def get_dataset(key):
    urls = {
        "one-source":"https://figshare.com/ndownloader/files/26648885",
        "hoax":"https://figshare.com/ndownloader/files/26648861"
    }
    response = requests.get(urls[key], allow_redirects=True)
    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    return df

def build_model(key, DROP_COLS=None):
    df = get_dataset(key)
    df.set_index('page_id', inplace=True)
    drop_colnames= ['revision_id.key', 'revision_id']
    if DROP_COLS:
        drop_colnames.extend(DROP_COLS)
    df.drop(drop_colnames, axis=1, inplace=True)

    # Encode any categorical variables
    df[['has_template']] = df[['has_template']].astype('bool')

    # Encode article quality score
    aq_score_map = {'article_quality_score':
                    {"List":0, "Stub": 1, "Start": 2, "C": 3, "B": 4, "GA": 5, "A": 6, "FL": 7, "FA": 8}
                   }
    df.replace(aq_score_map, inplace=True)

    # Assign label
    df.fillna(0, inplace=True)
    y = df.pop('has_template')
    X = df.copy()
    pageID_group = df.index.values


    # Standard scale our data
    feature_names = list(X.columns) #Keep track of feature names
    ss = StandardScaler()
    ss.fit(X)
    X = ss.transform(X)

    # Score dicts to keep track of scores across models
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    model_scores =  {}
    XGB_model = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
    XGB_scores = cross_validate(XGB_model, X, y, scoring=scoring, cv=GroupKFold(), groups=pageID_group)
    XGB_model.fit(X, y)
    print(TEMPLATE_NAME)
    print(XGB_scores)
    model_scores['XGB'] = XGB_scores
    XGB_model.get_booster().feature_names = feature_names
    feature_importance = XGB_model.get_booster().get_score(importance_type='gain')
    model_results = expand_scores(model_scores)
    return model_results, feature_importance, XGB_model, X, y