import numpy as np

def Gini_index(Y_data):
    gini = 0
    #=========    Edit here    ==========
    if isinstance(Y_data, (list, )):                # Gini 인덱스는 entropy랑 비슷하고 공식만 다르니까 설명 안할게.
        Y_data = np.asarray(Y_data)

    if isinstance(Y_data, np.ndarray):
        total_cnt = len(Y_data)
        unique, counts = np.unique(Y_data, return_counts=True)
        for cnt in counts:
            gini += (cnt/total_cnt)**2
    else:
        value_cnt = Y_data.value_counts()
        total_cnt = len(Y_data)

        for cnt in value_cnt:
            gini += (cnt/total_cnt)**2

    gini = 1 - gini

    #====================================
    return gini

def Entropy(Y_data):

    entropy = 0
    # =====    Edit here    ========
    if isinstance(Y_data, (list, )):                # 이게 나는 들어오는 인풋 타입이 다 달라서 나눠줌. 이건 리스트일때,
        Y_data = np.asarray(Y_data)                 # 그냥 ndarray로 만들어버리는 거임

    if isinstance(Y_data, np.ndarray):              # 이건 ndarray일 때, 아래 구조는 else문이랑 같음
        total_cnt = len(Y_data)                     # Y_data는 Yes, Yes, No, Yes 이런 식인 거임. 총 개수를 구함.
        unique, counts = np.unique(Y_data, return_counts=True)          # 이렇게하면 (Yes, 3) (No, 1) 이런 식으로 나뉨
        for cnt in counts:                                              # 아래 두식은 이제 엔트로피 구하는거 보면 알거임
            entropy += -(cnt/total_cnt)*np.log2(cnt/total_cnt)

    else:                                           # 여기도 똑같음. 이건 Pandas의 series로 인풋타입이 들어올때임
        value_cnt = Y_data.value_counts()           # series는 이런식으로 얻을 수 있더라. 아래는 똑같음.
        total_cnt = len(Y_data)
        for cnt in value_cnt:
            entropy += -(cnt/total_cnt)*np.log2(cnt/total_cnt)

    # ==============================
    return entropy

def impurity_func(Y_data, criterion):

    if criterion == 'gini':
        return Gini_index(Y_data)

    elif criterion == 'entropy':
        return Entropy(Y_data)

def Finding_split_point(df, feature, criterion):

    col_data = df[feature]
    Y = df.values[:, -1]
    distinct_data = np.unique(col_data)

    split_point = distinct_data[0]
    min_purity = 1

    for idx, val in enumerate(distinct_data):
        less_idx = (col_data < val)

        y0 = Y[less_idx]
        y1 = Y[~less_idx]

        p0 = len(y0) / len(Y)
        p1 = len(y1) / len(Y)

        purity = np.sum([p0 * impurity_func(y0, criterion), p1 * impurity_func(y1, criterion)])

        if min_purity > purity:
            min_purity = purity
            split_point = val

    return split_point

def Gaussian_prob(x, mean, std):
    '''
    :param x: input value: X
    :param mean: the mean of X
    :param std: the standard deviation of X
    :return: probaility (X) ~ N(μ, σ^2)
    '''
    ret = 0
    # ========      Edit here         ==========
    ret = 1 / (std * np.sqrt(2 * np.pi)) *  np.exp(- (x - mean) ** 2 / (2 * std ** 2))          # Gaussian_prob 구하는 식임. 이건 그냥 복붙해도됨. 나도 그냥 인터넷 복붙함.
    # =========================================
    return ret