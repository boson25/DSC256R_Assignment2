import numpy as np
from tqdm import tqdm

class MatrixFactorization:
    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.1, epoch=10):
        """
        num_users: total number of users
        num_items: total number of items
        k: latent dimension
        lr: learning rate
        reg: regularization strength (lamdba)
        epochs: number of SGD passes
        """

        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epoch = epoch

        # Initialize latent factors (P for users, Q for items)
        self.P = np.random.normal(0, 0.1, (num_users, k))
        self.Q = np.random.normal(0, 0.1, (num_items, k))

    def predict(self, user, item):
        """Predict rating for user-item pair."""
        return np.dot(self.P[user], self.Q[item])

    def fit(self, train_data):
        """
        train_data: list of (user, item, rating)
        SGD Optimization
        """

        for ep in range(self.epoch):
            np.random.shuffle(train_data)
            total_loss = 0

            for (u, i, r) in train_data:
                pred = self.predict(u, i)
                err = r - pred

                #Compute gradients and update latent factors
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] +  self.lr * (err * self.P[u] - self.reg * self.Q[i])

                total_loss += err**2 + self.reg * (np.linalg.norm(self.P[u])**2 + np.linalg.norm(self.Q[i])**2)

            print(f"[Epoch {ep+1}/{self.epoch}] Loss: {total_loss:.4f}")

    def evaluate(self, test_data):
        """Compute MSE on test set"""
        errors = []
        for (u, i, r) in test_data:
            pred = self.predict(u, i)
            errors.append((r - pred)**2)
        return np.mean(errors)