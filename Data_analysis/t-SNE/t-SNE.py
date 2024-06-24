from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# data load
digits = load_digits()

# t-SNE 모델 생성
tsne = TSNE(random_state=0) #seed를 고정함으로 결과를 같게 만듦
digits_tsne = tsne.fit_transform(digits.data) # digits.data를 입력으로 tsne를 사용

plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:,0].min(), digits_tsne[:,0].max()+1)
plt.ylim(digits_tsne[:,1].min(), digits_tsne[:,1].max()+1)

#시각화를 위한 10개의 색상 설정
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]


# 시각화
for i in range(len(digits.data)): # 0부터  digits.data까지 정수

    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), # x, y , 그룹

             color= colors[digits.target[i]], # 색상 지정

             fontdict={'weight': 'bold', 'size':9})
    
plt.show()