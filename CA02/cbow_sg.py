import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.special import softmax
from numpy.lib.stride_tricks import sliding_window_view


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

words = word_tokenize(text.lower())
words = [word for word in words if word.isalpha()]

vocab = sorted(set(words))
w2i = {word: i for i, word in enumerate(vocab)}

V = len(vocab)
D = EMBEDDING_SIZE


# 2. CREATE TRAINING PAIRS
windows = sliding_window_view(
    np.array(words),
    2 * WINDOW_SIZE + 1
)

# CBOW: context -> target
cbow = [
    (
        [w2i[w] for j, w in enumerate(window) if j != WINDOW_SIZE],
        w2i[window[WINDOW_SIZE]]
    )
    for window in windows
]

# Skip-gram: target -> context
skipgram = [
    (
        w2i[window[WINDOW_SIZE]],
        w2i[w]
    )
    for window in windows
    for j, w in enumerate(window)
    if j != WINDOW_SIZE
]


# 3. TRAIN
def train_word2vec(training_pairs, is_cbow):
    input_weights = np.random.randn(V, D) * 0.01
    output_weights = np.random.randn(D, V) * 0.01

    for epoch in range(EPOCHS):
        for input_data, target_index in training_pairs:

            if is_cbow:
                hidden = input_weights[input_data].mean(axis=0)
            else:
                hidden = input_weights[input_data]

            probabilities = softmax(hidden @ output_weights)

            gradient = probabilities.copy()
            gradient[target_index] -= 1

            output_gradient = np.outer(hidden, gradient)
            hidden_gradient = output_weights @ gradient

            if is_cbow:
                input_weights[input_data] -= (
                    LEARNING_RATE * hidden_gradient / len(input_data)
                )
            else:
                input_weights[input_data] -= (
                    LEARNING_RATE * hidden_gradient
                )

            output_weights -= LEARNING_RATE * output_gradient

    return input_weights


# 4. TRAIN CBOW AND SKIP-GRAM
cbow_vectors = train_word2vec(cbow, True)
skipgram_vectors = train_word2vec(skipgram, False)


# 5. SIMILARITY
word_a = w2i["king"]
word_b = w2i["queen"]

cbow_a = cbow_vectors[word_a].reshape(1, -1)
cbow_b = cbow_vectors[word_b].reshape(1, -1)

skip_a = skipgram_vectors[word_a].reshape(1, -1)
skip_b = skipgram_vectors[word_b].reshape(1, -1)


print("CBOW Cosine Similarity:",
      cosine_similarity(cbow_a, cbow_b)[0, 0])

print("CBOW Euclidean Distance:",
      euclidean_distances(cbow_a, cbow_b)[0, 0])

print("Skip-gram Cosine Similarity:",
      cosine_similarity(skip_a, skip_b)[0, 0])

print("Skip-gram Euclidean Distance:",
      euclidean_distances(skip_a, skip_b)[0, 0])
