import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#필요한 라이브러리 임포트
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy.linalg import norm

# 1. Skip-gram 모델 정의
class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.linear(embeddings)
        return output


# 2. 데이터 전처리
# 코퍼스 로드 (SSL 문제 해결 후 작동)
Korpora.fetch('nsmc')
corpus = Korpora.load("nsmc")
corpus = pd.DataFrame(corpus.test)

# 토크나이저 설정
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(tokens[:3])


# 3. 단어 사전 구축
def build_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab


# 단어 사전 생성
vocab = build_vocab(corpus=tokens, n_vocab=5000, special_tokens=["<unk>"])
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print(vocab[:10])
print(f"Vocabulary size: {len(vocab)}")


#4. Skip-gram 훈련 쌍 생성
def get_word_pairs(tokens, window_size):
    pairs = []
    for sentence in tokens:
        sentence_length = len(sentence)
        for idx, center_word in enumerate(sentence):
            window_start = max(0, idx - window_size)
            window_end = min(sentence_length, idx + window_size + 1)
            center_word = sentence[idx]
            context_words = sentence[window_start:idx] + sentence[idx + 1:window_end]
            for context_word in context_words:
                pairs.append([center_word, context_word])
    return pairs


word_pairs = get_word_pairs(tokens, window_size=2)
print(word_pairs[:5])


# 5. 인덱스 쌍 변환
def get_index_pairs(word_pairs, token_to_id):
    pairs = []
    unk_index = token_to_id["<unk>"]
    for word_pair in word_pairs:
        center_word, context_word = word_pair
        center_index = token_to_id.get(center_word, unk_index)
        context_index = token_to_id.get(context_word, unk_index)
        pairs.append([center_index, context_index])
    return pairs


index_pairs = get_index_pairs(word_pairs, token_to_id)
print(index_pairs[:5])


#  6. PyTorch TensorDataset 생성
index_pairs = torch.tensor(index_pairs)
center_indexes = index_pairs[:, 0]
context_indexes = index_pairs[:, 1]

dataset = TensorDataset(center_indexes, context_indexes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 7. 모델 학습 설정 (MPS 설정 적용)
device = "mps" if torch.backends.mps.is_available() else "cpu"
word2vec = VanillaSkipgram(vocab_size=len(token_to_id), embedding_dim=128).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(word2vec.parameters(), lr=0.1)

print(f"Using device: {device}")


# 8. 모델 학습
for epoch in range(10):
    cost = 0.0
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device).long()

        logits = word2vec(input_ids)
        loss = criterion(logits, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item()

    cost = cost / len(dataloader)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")


# 9. 임베딩 추출
token_to_embedding = dict()
embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()

for word, embedding in zip(vocab, embedding_matrix):
    token_to_embedding[word] = embedding

# 특정 토큰 확인
index = 30
token = vocab[index]
token_embedding = token_to_embedding[token]
print(f"Token: {token}")
print(f"Embedding: {token_embedding}")


# 10. 유사한 단어 찾기
def cosine_similarity(a, b):
    cosine = np.dot(b, a) / (norm(b, axis=1) * norm(a))
    return cosine


def top_n_index(cosine_matrix, n):
    closest_indexes = cosine_matrix.argsort()[::-1]
    top_n = closest_indexes[1 : n + 1]
    return top_n


# 코사인 유사도 계산
cosine_matrix = cosine_similarity(token_embedding, embedding_matrix)
top_n = top_n_index(cosine_matrix, n=5)

# 유사 단어 출력
print(f"\n[{token}]와 가장 유사한 5개 단어:")
for index in top_n:
    print(f"{id_to_token[index]} - 유사도 : {cosine_matrix[index]:.4f}")


# SSL 설정 복구 (필요한 경우)
ssl._create_default_https_context = ssl.create_default_context