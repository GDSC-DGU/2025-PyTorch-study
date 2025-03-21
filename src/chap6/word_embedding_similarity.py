import numpy as np
import torch
from torch import nn, optim
from numpy.linalg import norm

vocab = ["단어1", "단어2", "단어3", "단어4", "단어5", "사과", "배", "포도", "딸기", "오렌지",
         "컴퓨터", "키보드", "마우스", "모니터", "노트북", "프로그램", "코딩", "알고리즘", "데이터",
         "인공지능", "머신러닝", "딥러닝", "신경망", "자연어처리", "컴퓨터비전", "강화학습", "파이썬",
         "텐서플로우", "파이토치", "케라스", "넘파이"]
token_to_id = {token: idx for idx, token in enumerate(vocab)}  # 토큰을 ID로 매핑
id_to_token = {idx: token for idx, token in enumerate(vocab)}  # ID를 토큰으로 매핑

class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(VanillaSkipgram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 임베딩 레이어 초기화
        self.output_layer = nn.Linear(embedding_dim, vocab_size)  # 출력 레이어 초기화
    
    def forward(self, x):
        x = self.embedding(x)  # 입력 토큰을 임베딩으로 변환
        x = self.output_layer(x)  # 임베딩을 출력 레이어에 통과시켜 예측 생성
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"  # 사용 가능한 장치 설정
print(f"Using device: {device}")

word2vec = VanillaSkipgram(vocab_size=len(token_to_id), embedding_dim=128).to(device)  # 모델 초기화
criterion = nn.CrossEntropyLoss().to(device)  # 손실 함수 정의
optimizer = optim.SGD(word2vec.parameters(), lr=0.1)  # 옵티마이저 설정

token_to_embedding = dict()  # 단어와 임베딩 매핑을 위한 딕셔너리 초기화
embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()  # 임베딩 매트릭스 추출

for word, embedding in zip(vocab, embedding_matrix):  # 각 단어와 임베딩 매핑
    token_to_embedding[word] = embedding

index = min(30, len(vocab) - 1)  # 인덱스 범위 체크
token = vocab[index]  # 선택된 단어
token_embedding = token_to_embedding[token]  # 선택된 단어의 임베딩 추출
print(f"선택된 단어: {token}")
print(f"임베딩 벡터: {token_embedding[:5]}...")  # 임베딩 벡터의 처음 5개 요소 출력

def cosine_similarity(query_vector, matrix):
    dot_products = np.dot(matrix, query_vector)  # 벡터의 내적 계산
    matrix_norms = norm(matrix, axis=1)  # 행렬의 각 벡터 노름 계산
    query_norm = norm(query_vector)  # 쿼리 벡터의 노름 계산
    cosine = dot_products / (matrix_norms * query_norm)  # 코사인 유사도 계산
    return cosine

def top_n_index(cosine_matrix, n=5, exclude_self=True):
    closest_indexes = cosine_matrix.argsort()[::-1]  # 코사인 유사도를 내림차순으로 정렬한 인덱스
    start_idx = 1 if exclude_self else 0  # 자기 자신 제외 여부 결정
    top_n = closest_indexes[start_idx:start_idx + n]  # 상위 n개의 인덱스 선택
    return top_n

cosine_matrix = cosine_similarity(token_embedding, embedding_matrix)  # 코사인 유사도 계산

top_n = top_n_index(cosine_matrix, n=5)  # 상위 5개 유사 단어 찾기

print(f"\n'{token}'와 가장 유사한 5개 단어:")  # 결과 출력
for index in top_n:
    print(f"{vocab[index]} - 유사도: {cosine_matrix[index]:.4f}")  # 각 단어와 유사도 출력