import numpy as np
import re

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

words = re.sub(r'[^a-z ]', '', text.lower()).split()
vocab = sorted(set(words))
w2i = {w:i for i,w in enumerate(vocab)}
i2w = {i:w for w,i in w2i.items()}
V, D = len(vocab), 20


# 2. CREATE TRAINING PAIRS
window = 2
cbow, skipgram = [], []

for i in range(window, len(words)-window):
    ctx = [w2i[w] for w in words[i-window:i] + words[i+1:i+window+1]]
    tgt = w2i[words[i]]
    cbow.append((ctx, tgt))

for i, w in enumerate(words):
    for j in range(max(0,i-window), min(len(words),i+window+1)):
        if i != j:
            skipgram.append((w2i[w], w2i[words[j]]))


# 3. SOFTMAX
def softmax(x):
    e = np.exp(x - max(x))
    return e / e.sum()


# 4. TRAIN CBOW
W1 = np.random.randn(V,D) * .01
W2 = np.random.randn(D,V) * .01

for epoch in range(2000):
    for ctx, tgt in cbow:

        h = W1[ctx].mean(0)
        p = softmax(h @ W2)

        grad = p.copy()
        grad[tgt] -= 1

        dW2 = np.outer(h, grad)
        dh = W2 @ grad

        W1[ctx] -= .05 * dh / len(ctx)
        W2 -= .05 * dW2

cbow_vec = W1


# 5. TRAIN SKIP-GRAM
W1 = np.random.randn(V,D) * .01
W2 = np.random.randn(D,V) * .01

for epoch in range(2000):
    for tgt, ctx in skipgram:

        h = W1[tgt]
        p = softmax(h @ W2)

        grad = p.copy()
        grad[ctx] -= 1

        dW2 = np.outer(h, grad)
        dh = W2 @ grad

        W1[tgt] -= .05 * dh
        W2 -= .05 * dW2

skip_vec = W1


# 6. SIMILARITY
def sim(a,b,vec):
    x,y = vec[w2i[a]], vec[w2i[b]]
    return x @ y / (np.linalg.norm(x)*np.linalg.norm(y))

print("CBOW:", sim("king","queen",cbow_vec))
print("Skipgram:", sim("king","queen",skip_vec))
