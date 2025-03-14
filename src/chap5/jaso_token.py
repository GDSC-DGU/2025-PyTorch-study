from jamo import h2j, j2hcj

review = "빽그람핫도그"

decomposed = j2hcj(h2j(review))
tokenized = list(decomposed)
print(tokenized)
print(decomposed)