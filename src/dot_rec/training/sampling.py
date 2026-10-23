import numpy as np
from dot_rec.evaluation import histories


class TrainNegativeSampler:
    """Uses only TRAIN histories, with finite fallback for dense users."""
    def __init__(self, train_edges, n_items):
        self.history = histories(train_edges)
        self.n_items = n_items
        if any(len(items) >= n_items for items in self.history.values()):
            raise ValueError("A TRAIN user has no negative item in the catalog")

    def sample(self, users, rng):
        negatives = []
        for user in users:
            seen = self.history.get(int(user), set())
            for _ in range(64):
                item = int(rng.integers(self.n_items))
                if item not in seen:
                    break
            else:
                available = np.array([j for j in range(self.n_items) if j not in seen])
                item = int(rng.choice(available))
            negatives.append(item)
        return np.asarray(negatives, dtype=np.int64)
