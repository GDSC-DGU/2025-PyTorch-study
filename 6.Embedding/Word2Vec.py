# Word2Vec
# 어떤 단어 주변에 나타나는 단어들의 분포가 비슷하면, 단어는 유사한 의미를 가질 확률이 높다.
# 단어를 고정된 크기의 실수 벡터로 표현 -> Word Embedding Vector
import os
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from gensim.models import Word2Vec

# 데이터 로드 및 전처리
print("NSMC 코퍼스 로딩 중...")
Korpora.fetch('nsmc')
corpus = Korpora.load("nsmc")
corpus = pd.DataFrame(corpus.test)

# 토크나이저 설정
print("텍스트 토큰화 진행 중...")
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(f"토큰화 완료 - 총 샘플 수: {len(tokens)}")

# Word2Vec 학습 설정
print("Word2Vec 모델 학습 시작...")
word2vec = Word2Vec(
    sentences=tokens,           # 토큰화된 문장
    vector_size=128,            # 임베딩 벡터 차원 수
    window=5,                   # 윈도우 크기
    min_count=1,                # 최소 등장 횟수
    sg=1,                       # Skip-Gram 사용 (1: Skip-Gram, 0: CBOW)
    workers=os.cpu_count(),     # CPU 코어 수에 맞게 병렬 처리
)

# 학습 완료 및 모델 저장
model_path = os.path.join(os.getcwd(), "word2vec.model")
word2vec.save(model_path)
print(f"모델 저장 완료 → {model_path}")

# 모델 로드
word2vec = Word2Vec.load(model_path)
print("모델 로드 완료")

# 특정 단어 임베딩 출력
word = "연기"
if word in word2vec.wv:
    print(f"\n'{word}'의 벡터:\n{word2vec.wv[word]}")
    print(f"\n'{word}'와 유사한 단어:")
    similar_words = word2vec.wv.most_similar(word, topn=5)
    for similar_word, similarity in similar_words:
        print(f"  {similar_word} - 유사도: {similarity:.4f}")

    # 두 단어 간 유사도 계산
    similarity = word2vec.wv.similarity(w1=word, w2="연기력")
    print(f"\n'{word}'와 '연기력'의 유사도: {similarity:.4f}")
else:
    print(f"'{word}'가 어휘에 없습니다.")