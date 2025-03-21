# fastText OOV
import os
from Korpora import Korpora
from gensim.models import FastText

# 데이터 로드 및 전처리
print("KoNLI 코퍼스 로딩 중...")
Korpora.fetch('kornli')  # KoNLI 코퍼스 다운로드
corpus = Korpora.load("kornli")

# KoNLI 데이터에서 텍스트 추출
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()

# 텍스트를 토큰화
tokens = [sentence.split() for sentence in corpus_texts]
print(f"토큰화 완료 - 총 샘플 수: {len(tokens)}")

# FastText 모델 학습 설정
print("FastText 모델 학습 시작...")
fastText = FastText(
    sentences=tokens,           # 입력 텍스트 (토큰화된 문장 리스트)
    vector_size=128,            # 임베딩 벡터 차원 수
    window=5,                   # 윈도우 크기
    min_count=5,                # 최소 등장 횟수
    sg=1,                       # Skip-Gram 사용 (1: Skip-Gram, 0: CBOW)
    max_final_vocab=20000,      # 최대 어휘 수
    epochs=3,                   # 학습 반복 횟수
    min_n=2,                    # FastText 하위 문자열 최소 길이
    max_n=6,                    # FastText 하위 문자열 최대 길이
    workers=os.cpu_count()      # 병렬 학습 설정
)

# 모델 저장
model_path = os.path.join(os.getcwd(), "fastText.model")
fastText.save(model_path)
print(f"모델 저장 완료 → {model_path}")

# 모델 로드
fastText = FastText.load(model_path)
print("모델 로드 완료")

# OOV 단어 벡터 출력 (FastText는 서브워드 학습을 통해 OOV 처리 가능)
oov_token = "사랑해요"

# FastText는 서브워드로 학습했기 때문에 OOV 벡터 생성 가능
if oov_token in fastText.wv.index_to_key:
    print(f"\n'{oov_token}'의 벡터:\n{fastText.wv[oov_token]}")
else:
    print(f"\n'{oov_token}'은 어휘에 없지만 서브워드 벡터 생성 가능")

# OOV 벡터를 통한 유사 단어 검색
oov_vector = fastText.wv[oov_token]
print(f"\n'{oov_token}'와 유사한 단어:")
similar_words = fastText.wv.most_similar(oov_vector, topn=5)
for similar_word, similarity in similar_words:
    print(f"  {similar_word} - 유사도: {similarity:.4f}")