from gensim.models import Word2Vec

# 학습용 말뭉치 (여기선 vocab을 문장 형태로 간단히 구성)
corpus = [
    ["사과", "배", "포도", "딸기", "오렌지"],
    ["컴퓨터", "키보드", "마우스", "모니터", "노트북"],
    ["프로그램", "코딩", "알고리즘", "데이터"],
    ["인공지능", "머신러닝", "딥러닝", "신경망", "자연어처리", "컴퓨터비전", "강화학습"],
    ["파이썬", "텐서플로우", "파이토치", "케라스", "넘파이"]
]

# Word2Vec 모델 학습
model = Word2Vec(
    sentences=corpus,
    vector_size=128,  # 임베딩 차원
    window=2,         # 주변 단어 윈도우 크기
    min_count=1,      # 최소 등장 빈도
    sg=1,             # Skip-gram 사용 (0이면 CBOW)
    epochs=100        # 에폭 수
)

# 테스트할 단어 선택
token = "파이토치"
print(f"선택된 단어: {token}")
print(f"임베딩 벡터 (처음 5차원): {model.wv[token][:5]}...")

# 유사한 단어 Top 5 출력
top_n = model.wv.most_similar(token, topn=5)
print(f"\n'{token}'와 가장 유사한 5개 단어:")
for word, similarity in top_n:
    print(f"{word} - 유사도: {similarity:.4f}")