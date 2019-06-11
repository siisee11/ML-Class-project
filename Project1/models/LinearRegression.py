import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Training should be done for 'epochs' times with minibatch size of 'batch_size'
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

                estimated_y = batch_x.dot(w)
                loss = np.reshape((batch_y - estimated_y), (1, -1))
                grad = np.reshape(-(1.0 / len(batch_x)) * loss.dot(batch_x), (-1, 1))
                w = optim.update(w, grad, lr)
                loss = loss.mean()
                final_loss = np.power(loss, 2)
                print(final_loss)

        self.W = w

        # ============================================================
        return final_loss

    def eval(self, x):
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        pred = x.dot(self.W)


        # ============================================================
        return pred
