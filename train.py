from model import MLP
from dataset import train_loader,valid_loader,test_loader
from tqdm import tqdm
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0005)

#学習
print("training start ----")
num_epochs = 50

#学習コード

def train_model(model,loader,criterion,optimizer,device):
    model.train()
    total_loss, correct = 0.0,0

    pbar = tqdm(loader,desc = "Training")
    for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch) #クロスエントロピーが平均を返しているので、長さでかけて全体のロスへ戻す
            correct += (output.argmax(dim=1) == y_batch).sum().item() #outputの中身は２つの値でargmaxは数が大きい方のインデックス番号を返す。dim = 1はリスト全体の横方向でargmaxを返す意味。まだybatchはテンソルなので、数値へ戻す

            pbar.set_postfix(loss = loss.item())

    n = len(loader.dataset)#loaderだけだとバッチ数になってしまうので、(分けた体の集合体だから)データの総数を取る.dataset
    return total_loss / n, correct / n
    
#評価コード、学習モデルと一緒にやることによって過学習を防ぐ
@torch.no_grad()
def evaluate(model,loader,criterion,device):
      model.eval()
      total_loss, correct = 0.0,0

      for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            loss = criterion(output,y_batch)

            total_loss += loss.item() * len(y_batch)
            correct += (output.argmax(dim = 1) == y_batch).sum().item()

      n = len(loader.dataset)
      return total_loss / n , correct / n

for epoch in tqdm(range(1, num_epochs + 1), desc = "Epoch"):
      train_loss, train_acc = train_model(model,train_loader,criterion,optimizer,device)
      valid_loss, valid_acc = evaluate(model, valid_loader,criterion, device)
      print(f"Epoch{epoch} |"
            f"Train Loss : {train_loss:.4f} Train Acc : {train_acc:.4f}"
            f" Valid Loss : {valid_loss:.4f} Valid acc : {valid_acc:.4f}")

