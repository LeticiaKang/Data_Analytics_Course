
def notDivide_Standard(model_name, X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import StandardScaler 
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import Normalizer
    # from sklearn.svm import SVC
    StandardScaler = StandardScaler()
    RobustScaler = RobustScaler()
    MinMaxScaler = MinMaxScaler()
    Normalizer = Normalizer()
    # svc = SVC()
    
    for s in [Normalizer, RobustScaler, MinMaxScaler, Normalizer]:
        s.fit(X_train)
        X_train_scaled = s.transform(X_train)
        X_test_scaled = s.transform(X_test)
        model_name.fit(X_train_scaled, y_train)
        print(f"============== {s} 스케일러 ===============")
        print("학습데이터 정확도 :", model_name.score(X_train_scaled, y_train))
        print("테스트데이터 정확도 :", model_name.score(X_test_scaled, y_test))
        print("")
        
        
        
        
        
def YD_score(modelName, X_train, X_test, y_train, y_test):
    modelName.fit(X_train, y_train)        
    print(f"============== {modelName}의 정확도===============")
    print("학습데이터 정확도 :", modelName.score(X_train, y_train))
    print("테스트데이터 정확도 :", modelName.score(X_test, y_test))
    print("")
    




## 원하는 스케일러를 불러오고 싶을 때, 데이터셋을 나누지 않았을 경우
def ND_Standard(X, y, want_randomtate):
    from sklearn.preprocessing import StandardScaler 
    from sklearn.model_selection import train_test_split
    StandardScaler = StandardScaler()
    # 스케일러로 한 번에 변황
    X_scaled = StandardScaler.fit_transform(X)
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=want_randomtate)
    # 스케일링 된 데이터를 나눈 것
    return X_train, X_test, y_train, y_test
    
    
    
    
def ND_Robust(X, y, want_randomtate):
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    RobustScaler = RobustScaler()
    # 스케일러로 한 번에 변황
    X_scaled = RobustScaler.fit_transform(X)
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=f"{want_randomState}")
    # 스케일링 된 데이터를 나눈 것
    return X_train, X_test, y_train, y_test
          
          
          
          
def ND_MinMax(X, y, want_randomtate):
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.model_selection import train_test_split
    MinMaxScaler = MinMaxScaler()
    # 스케일러로 한 번에 변황
    X_scaled = MinMaxScaler.fit_transform(X)
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=f"{want_randomState}")
    # 스케일링 된 데이터를 나눈 것
    return X_train, X_test, y_train, y_test
    
    
    
def ND_Normal(X, y, want_randomtate):
    from sklearn.preprocessing import Normalizer
    from sklearn.model_selection import train_test_split
    Normalizer = Normalizer()
    # 스케일러로 한 번에 변황
    X_scaled = Normalizer.fit_transform(X)
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=f"{want_randomState}")
    # 스케일링 된 데이터를 나눈 것
    return X_train, X_test, y_train, y_test
    
    



