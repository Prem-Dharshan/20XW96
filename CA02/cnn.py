import numpy as np
import re

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# Variables
EMBEDDING_SIZE = 20
WINDOW_SIZE = 2
EPOCHS = 2000
LEARNING_RATE = 0.05


# 1. PREPROCESS
text = """
king queen king queen
king is a man
queen is a woman
king rules kingdom
queen rules kingdom
king and queen rule kingdom
man is human
woman is human
king is powerful
queen is powerful
king is royal
queen is royal
"""

words = re.sub(r"[^a-z ]", "", text.lower()).split()

vocab = sorted(set(words))
w2i = {word: i for i, word in enumerate(vocab)}
i2w = {i: word for word, i in w2i.items()}

V = len(vocab)
D = EMBEDDING_SIZE


# 2. CREATE TRAINING PAIRS

cbow = []
skipgram = []

for i in range(WINDOW_SIZE, len(words) - WINDOW_SIZE):
    context = [
        w2i[word]
        for word in words[i-WINDOW_SIZE:i] +
                   words[i+1:i+WINDOW_SIZE+1]
    ]
    target = w2i[words[i]]
    cbow.append((context, target))


for i, word in enumerate(words):
    for j in range(
        max(0, i-WINDOW_SIZE),
        min(len(words), i+WINDOW_SIZE+1)
    ):
        if i != j:
            skipgram.append(
                (w2i[word], w2i[words[j]])
            )


# 3. SOFTMAX
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


# 4. TRAIN CBOW

W1 = np.random.randn(V, D) * 0.01
W2 = np.random.randn(D, V) * 0.01

for epoch in range(EPOCHS):
    for context, target in cbow:

        h = W1[context].mean(axis=0)
        p = softmax(h @ W2)

        grad = p.copy()
        grad[target] -= 1

        dW2 = np.outer(h, grad)
        dh = W2 @ grad

        W1[context] -= LEARNING_RATE * dh / len(context)
        W2 -= LEARNING_RATE * dW2

cbow_vec = W1


# 5. TRAIN SKIP-GRAM

W1 = np.random.randn(V, D) * 0.01
W2 = np.random.randn(D, V) * 0.01

for epoch in range(EPOCHS):
    for target, context in skipgram:

        h = W1[target]
        p = softmax(h @ W2)

        grad = p.copy()
        grad[context] -= 1

        dW2 = np.outer(h, grad)
        dh = W2 @ grad

        W1[target] -= LEARNING_RATE * dh
        W2 -= LEARNING_RATE * dW2

skipgram_vec = W1


# 6. SIMILARITY

word_a = w2i["king"]
word_b = w2i["queen"]

cbow_a = cbow_vec[word_a].reshape(1, -1)
cbow_b = cbow_vec[word_b].reshape(1, -1)

skip_a = skipgram_vec[word_a].reshape(1, -1)
skip_b = skipgram_vec[word_b].reshape(1, -1)


print("CBOW Cosine Similarity:",
      cosine_similarity(cbow_a, cbow_b)[0, 0])

print("CBOW Euclidean Distance:",
      euclidean_distances(cbow_a, cbow_b)[0, 0])

print("Skip-gram Cosine Similarity:",
      cosine_similarity(skip_a, skip_b)[0, 0])

print("Skip-gram Euclidean Distance:",
      euclidean_distances(skip_a, skip_b)[0, 0])
