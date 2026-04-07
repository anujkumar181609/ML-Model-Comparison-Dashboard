#classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#regression models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error,r2_score

#scaling
from sklearn.preprocessing import StandardScaler


def train_models(X,y, problem_type):
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
    #scaling for linear regression
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    
    #ye dictionary hai to store every model's results
    results={}
    
    if problem_type =="classification":
        models= {
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10)
        }
    
        for name,model in models.items():
            if name=="SVM" or name=="KNN" or name=="Naive Bayes":
                model.fit(X_train_lr, y_train)
                y_pred= model.predict(X_test_lr)
                
            else:
                model.fit(X_train,y_train)
                y_pred= model.predict(X_test)
        
            acc= accuracy_score(y_test, y_pred)
            f1= f1_score(y_test, y_pred, average="weighted")
    
            results[name]={
                "accuracy": acc,
                "f1_score": f1,
                "score": f1
            }
        
    else:
        models={
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "Linear Regression": LinearRegression()
        }
        for name, model in models.items():
            
            if name =="Linear Regression":
                model.fit(X_train_lr, y_train)
                y_pred= model.predict(X_test_lr)
                
            else:
                model.fit(X_train, y_train)
                y_pred= model.predict(X_test)
        
            mae= mean_absolute_error(y_test,y_pred)
            r2= r2_score(y_test, y_pred)
        
            results[name]={
                "MAE":mae,
                "R2":r2,
                "score":r2
            }
    return results
    