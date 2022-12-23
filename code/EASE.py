import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# 딥 러닝 모델이 아니므로 nn.Module을 상속받을 필요가 없다. 
class EASE:
    def __init__(self, _lambda=0.5):
        self.B = None
        self._lambda = _lambda

    def train(self, X):
        G = (X.T @ X).toarray() # G = X'X
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self._lambda   # X'X + λI
        P = np.linalg.inv(G)    # P = (X'X + λI)^(-1)
        self.B = P / -np.diag(P)    # - P_{ij} / P_{jj} if i ≠ j
        self.B[diag_indices] = 0  # 대각행렬 원소만 0으로 만들어주기 위해

    def forward(self, user_row):
        return user_row @ self.B


df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
users = df['user'].unique()
items = df['item'].unique()

user2id = dict((user, id) for (id, user) in enumerate(users))
item2id = dict((item, id) for (id, item) in enumerate(items))
id2user = dict((id, user) for (id, user) in enumerate(users))
id2item = dict((id, item) for (id, item) in enumerate(items))

user_id = df['user'].apply(lambda x: user2id[x])
item_id = df['item'].apply(lambda x: item2id[x])
values = np.ones(df.shape[0])

X = csr_matrix((values, (users, items)))

model = EASE()
model.train(X)

result = -model.forward(X[:, :])
result[X.nonzero()] = np.inf  # 이미 어떤 한 유저가 클릭 또는 구매한 아이템 이력은 제외
result = result.argsort()[:,:10]

user_item = [(i,k) for i,j in enumerate(result) for k in j]
id_frame = pd.DataFrame(user_item, columns=['user','item'])
user_name = id_frame['user'].apply(lambda x: id2user[x])
item_name = id_frame['item'].apply(lambda x: id2item[x])

submit = pd.DataFrame(data={'user':user_name, 'item':item_name}, columns=['user','item'])
submit.to_csv('output/EASE_submit.csv', index=False)