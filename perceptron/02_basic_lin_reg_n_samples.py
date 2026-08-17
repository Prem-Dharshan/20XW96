import numpy as np

inputs = np.array([[1, 2, 4], [2, 3, 5], [3, 4, 6]])

weights = np.array([0.2, 0.8, -0.5])

bias = 2


def lin_reg_activation(x):
    return x


def predict(inputs, weights, bias):

    weighted_sum = np.sum(inputs * weights, axis=1)
    weighted_sum = inputs @ weights
    total_sum = weighted_sum + bias

    return lin_reg_activation(total_sum)


def main():
    prediction = predict(inputs, weights, bias)

    print(f"Prediction: {prediction}")


main()
