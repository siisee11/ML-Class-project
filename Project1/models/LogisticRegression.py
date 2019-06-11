import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================

        w = self.W

        for epoch in np.arange(0, epochs):
            for i in np.arange(0, x.shape[0], batch_size):
                # yield a tuple of the current batched data and labels
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                batch_y = np.reshape(batch_y, (-1, 1))

#                print("batch x shape : ", batch_x.shape)
#                print("batch y shape : ", batch_y.shape)

                ey = self._sigmoid(batch_x.dot(w))
#                print("w shape : ", w.shape)
#                print(ey)
                loss = np.reshape(-(batch_y * np.log(ey) + (1 - batch_y) * np.log(1 - ey)), (1, -1))
                diff = np.reshape(batch_y - ey, (1, -1))
                grad = np.reshape(-(1.0 / len(batch_x)) * diff.dot(batch_x), (-1, 1))
                w = optim.update(w, grad, lr)
                final_loss = np.power(loss.mean(), 2)

        self.W = w

        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        pred = x.dot(self.W)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(-x))
        # ============================================================
        return sigmoid