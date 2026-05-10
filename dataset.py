from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

print(data.feature_names)#dataの特徴量の名前を確認
print(data.data.shape)#データの大きさ
print(data.data[:5])#最初5個のデータの中

#学習用/検証用データセットへ分割

#x trainとy trainあるけど、x trainはもっかい7:3くらいで分けてvalid用にする。ここで分けた3割はtestに回す

X,Y = df.drop(columns=['target']),df['target']

#型確認

print(type(X),type(Y))

#型がpandas.DataFrame, pandas.Seriesだったので、numpy配列に変換

X,Y = X.values,Y.values

#型確認

print(type(X),type(Y))
#無事numpy配列に変換


#分割、学習 : テスト = 7 : 3くらい

x_train, x_test, y_train, y_test = train_test_split(
    X,Y, test_size=0.3, random_state=42
)

#pytorch用データセットの変換

#学習用データセットの学習/検証の分割、同じく7 : 3くらい

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,y_train,test_size=0.3, random_state = 42
)

#train_test_split使うときは x 学習 x 検証/テスト y 学習 y検証/テスト　の順で書く

#全体の大きさの確認

print(X.shape,Y.shape)
print(x_train.shape,y_train.shape)#学習
print(x_valid.shape,y_valid.shape)#検証
print(x_test.shape,y_test.shape)#テスト

#テンソル変換

x_train_tensor = torch.tensor(x_train,dtype = torch.float32)
y_train_tensor = torch.tensor(y_train,dtype = torch.long)

x_valid_tensor = torch.tensor(x_valid,dtype = torch.float32)
y_valid_tensor = torch.tensor(y_valid,dtype = torch.long)

x_test_tensor = torch.tensor(x_test,dtype = torch.float32)
y_test_tensor = torch.tensor(y_test,dtype = torch.long)

#データとラベルの統合

train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
valid_dataset = TensorDataset(x_valid_tensor,y_valid_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

#DataLoader

#DataLoaderの役割って何って思ったけど、ミニバッチ学習とか、シャッフルとかしてくれるらしくて、そういう面で意味があるらしい

train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle = False)