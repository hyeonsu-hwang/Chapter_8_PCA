import numpy as np

A = np.matrix([[3,1,1], [-1,3,1]])

U, s, Vt = np.linalg.svd(A) # np의 선행대수 알고리즘 

V2 = Vt.T[:, :2]

A2 = A.dot(V2) # A행렬과 V2행렬의 곱

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1)
X, y = mnist["data"], mnist['target']

X_100 = X.head(100)
y_100 = y.head(100)

from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_PCA = pca.fit_transform(X)

X_PCA


var = pca.explained_variance_ratio_ # 설명된 분산의 비율

var_cum = var.cumsum() # 분산의 누적값

pca3 = PCA(n_components= 0.95) # n_componets에 0.0 에서 1.0사이로 설정하면 자동으로 해당 분산비율의 차원으로 차원축소를 진행함.
X_PCA3 = pca3.fit_transform(X_100)


## 과제
# MNIST 데이터셋을 로드하고 훈련데이터와 테스트데이터로 분할합니다.(처음 60000개는 훈련데이터, 나머지는 10000개는 테스트데이터)
# 이 데이터셋에 랜덤 포레스트 분류기로 훈련시키고 얼마나 오래 걸리는지 시간을 잰 다음, 테스트 데이터로 만들어진 모델을 평가해 정확도를 측정합니다.
# 그런다음 PCA를 사용해 설명된 분산이 95%가 되도록 차원을 축소합니다. 이 축소된 데이터셋에 새로운 랜덤 포레스트 분류기를 훈련시키고 얼마나 오래 걸리는지 확인하고
# 테스트 데이터로 모델을 평가해 정확도와 시간을 측정하시오.

## 2-1 차원축소하기 전의 훈련시간을 구하시오.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# 데이터 셋 dataframe에 저장
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1)
X, y = mnist["data"], mnist['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14285, random_state=5) # 랜덤으로 6만개 훈련데이터, 나머지 10000개는 데스트데이터로
y_train = y_train.values.ravel() # 랜덤포레스트 모델 학습시킬때 경고메세지가 안나오게 하는 코드

# 랜덤포레스트 모델 정의
RandomForestClassifier() # 비워두면 기본옵션들을 사용

# n_estimators -> 랜덤포레스트모델이 결정트리 몇 개를 만들어서 예측을 할지 정해주는 파라미터
# 안쓰면 기본값이 10이다
RandomForestClassifier(n_estimators=100)  # 100개의 결정트리를 사용하라

# max_depth -> 만들 트리들의 최대 깊이 정해주는 파라미터
RandomForestClassifier(max_depth=4)  # 최대 깊이를 4로 설정

model = RandomForestClassifier(n_estimators=100, max_depth=4)

# 모델 학습시키기
# model.fit(X_train, y_train)

import time
start = time.time()  # 시작 시간 저장
# 작업 코드
model.fit(X_train, y_train)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간


## 2-2 차원축소하기 전의 훈련데이터의 정확도를 구하시오.
# 분류모델 평가 -> 예측한 값들중에 몇 퍼센트가 맞게 분류됐는지 확인
model.score(X_train, y_train)

## 2-3 차원축소하기 전의 테스트데이터의 정확도를 구하시오.
model.score(X_test, y_test)


############################################################ 차원축소 후 ################################################################################################
## 2-4 차원축소하고 난 후의 훈련시간을 구하시오
from sklearn.decomposition import PCA
pca3 = PCA(n_components= 0.95) # n_componets에 0.0 에서 1.0사이로 설정하면 자동으로 해당 분산비율의 차원으로 차원축소를 진행함.
pca3.fit(X_train)
X_train_PCA = pca3.transform(X_train) # X_train_PCA = pca3.fit_transform(X_train) 이거랑 두줄 같다. 


# 랜덤포레스트 모델 정의
model = RandomForestClassifier(n_estimators=100, max_depth=4)

import time
start = time.time()  # 시작 시간 저장
# 작업 코드
model.fit(X_train_PCA, y_train)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

## 2-5 차원축소하고 난 후의 훈련데이터의 정확도를 구하시오.
# 분류모델 평가 -> 예측한 값들중에 몇 퍼센트가 맞게 분류됐는지 확인
model.score(X_train_PCA, y_train)

## 2-6 차원축소하고 난 후의 테스트데이터의 정확도를 구하시오.
X_test_PCA = pca3.transform(X_test)
model.score(X_test_PCA, y_test)


## score 메소드 VS accuracy_score 함수
# accuracy_score를 사용하기 위해서는 모델의 예측값이 필요했다.
# score 메소드는 클래스별로 다르게 정의되어 있었으며, 모델의 예측값을 별도로 구할 필요가 없었다.
# 분류 모델에서는 accuracy_score 함수와 score 메소드가 동일한 동작을 한다.

## transfrom() 메서드 vs fit_transform() 메서드
# fit_transform()은 train dataset에서만 사용됩니다
# train data로부터 학습된 mean값과 variance값을 test data에 적용하기 위해 transform() 메서드를 사용합니다
# 만약에 fit_transform을 test data에도 적용하게 된다면 test data로부터 새로운 mean값과 variance값을 얻게 되는 것입니다
# 즉, 우리의 모델이 test data도 학습하게 되는 것입니다


                                         

