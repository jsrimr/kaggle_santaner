import math
import io

# 파일 압축 용도
import gzip
import pickle
import zlib

# 데이터, 배열을 다루기 위한 기본 라이브러리
import pandas as pd
import numpy as np

# 범주형 데이터를 수치형으로 변환하기 위한 전처리 도구
from sklearn.preprocessing import LabelEncoder

import engines
from utils import *

def make_submission(f, Y_test, C):
    Y_ret = []
    # 파일의 첫 줄에 header를 쓴다
    f.write("ncodpers,added_products\n".encode('utf-8'))
    # 고객 식별 번호(C)와, 예측 결과물(Y_test)의 for loop
    for c, y_test in zip(C, Y_test):
        # (확률값, 금융 변수명, 금융 변수 id)의 tuple을 구한다
        y_prods = [(y,p,ip) for y,p,ip in zip(y_test, products, range(len(products)))]
        # 확률값을 기준으로 상위 7개 결과만 추출한다
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        # 금융 변수 id를 Y_ret에 저장한다
        Y_ret.append([ip for y,p,ip in y_prods])
        y_prods = [p for y,p,ip in y_prods]
        # 파일에 “고객 식별 번호, 7개의 금융 변수”를 쓴다
        f.write(("%s,%s\n" % (int(c), " ".join(y_prods))).encode('utf-8'))
    # 상위 7개 예측값을 반환한다
    return Y_ret

def train_predict(all_df, features, prod_features, str_date, cv):
    # all_df : 통합 데이터
    # features : 학습에 사용할 변수
    # prod_features : 24개 금융 변수
    # str_date : 예측 결과물을 산출하는 날짜. 2016-05-28일 경우, 훈련 데이터의 일부이며 정답을 알고 있기에 교차 검증을 의미하고, 2016-06-28일 경우, 캐글에 업로드하기 위한 테스트 데이터 예측 결과물을 생성한다
    # cv : 교차 검증 실행 여부

    # str_date로 예측 결과물을 산출하는 날짜를 지정한다
    test_date = date_to_int(str_date)
    # 훈련 데이터는 test_date 이전의 모든 데이터를 사용한다
    train_df = all_df[all_df.int_date < test_date]
    # 테스트 데이터를 통합 데이터에서 분리한다
    test_df = pd.DataFrame(all_df[all_df.int_date == test_date])

    # 신규 구매 고객만을 훈련 데이터로 추출한다
    X = []
    Y = []
    for i, prod in enumerate(products):
        prev = prod + "_prev1"
        # 신규 구매 고객을 prX에 저장한다
        prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
        # prY에는 신규 구매에 대한 label 값을 저장한다
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)

    XY = pd.concat(X)
    Y = np.hstack(Y)
    # XY는 신규 구매 데이터만 포함한다
    XY["y"] = Y

    # 메모리에서 변수 삭제
    del train_df
    del all_df

    # 데이터별 가중치를 계산하기 위해서 새로운 변수 (ncodpers + fecha_dato)를 생성한다
    XY["ncodepers_fecha_dato"] = XY["ncodpers"].astype(str) + XY["fecha_dato"]
    uniqs, counts = np.unique(XY["ncodepers_fecha_dato"], return_counts=True)
    # 자연 상수(e)를 통해서, count가 높은 데이터에 낮은 가중치를 준다
    weights = np.exp(1 / counts - 1)

    # 가중치를 XY 데이터에 추가한다
    wdf = pd.DataFrame()
    wdf["ncodepers_fecha_dato"] = uniqs
    wdf["counts"] = counts
    wdf["weight"] = weights
    XY = XY.merge(wdf, on="ncodepers_fecha_dato")

    print( " is XY empty? {}".format("Yes" if XY is None else "No"))

    # 교차 검증을 위하여 XY를 훈련:검증 (8:2)로 분리한다
    mask = np.random.rand(len(XY)) < 0.8
    XY_train = XY[mask]
    XY_validate = XY[~mask]

    # 테스트 데이터에서 가중치는 모두 1이다
    test_df["weight"] = np.ones(len(test_df), dtype=np.int8)

    # 테스트 데이터에서 “신규 구매” 정답값을 추출한다.
    test_df["y"] = test_df["ncodpers"]
    Y_prev = test_df.as_matrix(columns=prod_features)
    for prod in products:
        prev = prod + "_prev1"
        padd = prod + "_add"
        # 신규 구매 여부를 구한다
        test_df[padd] = test_df[prod] - test_df[prev]

    test_add_mat = test_df.as_matrix(columns=[prod + "_add" for prod in products])
    C = test_df.as_matrix(columns=["ncodpers"])
    test_add_list = [list() for i in range(len(C))]
    # 평가 척도 MAP@7 계산을 위하여, 고객별 신규 구매 정답값을 test_add_list에 기록한다
    count = 0
    for c in range(len(C)):
        for p in range(len(products)):
            if test_add_mat[c, p] > 0:
                test_add_list[c].append(p)
                count += 1

    # 교차 검증에서, 테스트 데이터로 분리된 데이터가 얻을 수 있는 최대 MAP@7 값을 계산한다.
    if cv:
        max_map7 = mapk(test_add_list, test_add_list, 7, 0.0)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        print("Max MAP@7", str_date, max_map7, max_map7 * map7coef)

    # LightGBM 모델 학습 후, 예측 결과물을 저장한다
    Y_test_lgbm = engines.lightgbm(XY_train, XY_validate, test_df, features, XY_all=XY,
                                   restore=(str_date == "2016-06-28"))
    test_add_list_lightgbm = make_submission(
        io.BytesIO() if cv else gzip.open("%s.lightgbm.csv.gz" % str_date, "wb"), Y_test_lgbm - Y_prev, C)

    # 교차 검증일 경우, LightGBM 모델의 테스트 데이터 MAP@7 평가 척도를 출력한다
    if cv:
        map7lightgbm = mapk(test_add_list, test_add_list_lightgbm, 7, 0.0)
        print("LightGBMlib MAP@7", str_date, map7lightgbm, map7lightgbm * map7coef)

    # XGBoost 모델 학습 후, 예측 결과물을 저장한다
    Y_test_xgb = engines.xgboost(XY_train, XY_validate, test_df, features, XY_all=XY,
                                 restore=(str_date == "2016-06-28"))
    test_add_list_xgboost = make_submission(io.BytesIO() if cv else gzip.open("%s.xgboost.csv.gz" % str_date, "wb"),
                                            Y_test_xgb - Y_prev, C)

    # 교차 검증일 경우, XGBoost 모델의 테스트 데이터 MAP@7 평가 척도를 출력한다
    if cv:
        map7xgboost = mapk(test_add_list, test_add_list_xgboost, 7, 0.0)
        print("XGBoost MAP@7", str_date, map7xgboost, map7xgboost * map7coef)

    # 곱셈 후, 제곱근을 구하는 방식으로 앙상블을 수행한다
    Y_test = np.sqrt(np.multiply(Y_test_xgb, Y_test_lgbm))
    # 앙상블 결과물을 저장하고, 테스트 데이터에 대한 MAP@7 를 출력한다
    test_add_list_xl = make_submission(
        io.BytesIO() if cv else gzip.open("%s.xgboost-lightgbm.csv.gz" % str_date, "wb"), Y_test - Y_prev, C)

    # 정답값인 test_add_list와 앙상블 모델의 예측값을 mapk 함수에 넣어, 평가 척도 점수를 확인한다
    if cv:
        map7xl = mapk(test_add_list, test_add_list_xl, 7, 0.0)
        print("XGBoost+LightGBM MAP@7", str_date, map7xl, map7xl * map7coef)

import os
if __name__ == "__main__":
    if os.path.exists("../input/8th.feature_engineer.all.pkl"):
        with open("../input/8th.feature_engineer.all.pkl", 'rb') as f:
            all_df = pickle.load(f)

        with open("../input/8th.feature_engineer.cv_meta.pkl", "rb") as f:
            features, prod_features = pickle.load(f)

        print("presaved df and features are loaded!")
    else:
        all_df, features, prod_features = make_data()
        # 피쳐 엔지니어링이 완료된 데이터를 저장한다
        all_df.to_pickle("../input/8th.feature_engineer.all.pkl")
        pickle.dump((features, prod_features), open("../input/8th.feature_engineer.cv_meta.pkl", "wb"))

        print("no presaved file exists. start from scratch")

    train_predict(all_df, features, prod_features, "2016-06-28", cv=False)
