import numpy as np
from typing import Dict

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

class Word2VecSGNS:
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            word2id: Dict[str, int],
            id2word: Dict[int, str],
            lr: float = 0.05,
            seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, size=(vocab_size, dim)).astype(np.float64)  # central embeddings
        self.U = rng.normal(0.0, 0.01, size=(vocab_size, dim)).astype(np.float64)  # context embeddings

        self.dV = np.zeros_like(self.V)
        self.dU = np.zeros_like(self.U)
        self.lr = lr
        self.cache = None
        self.word2id = word2id
        self.id2word = id2word

    def zero_grad(self):
        self.dV.fill(0.0)
        self.dU.fill(0.0)

    def forward(self, center_ids: np.ndarray, pos_ids: np.ndarray, neg_ids: np.ndarray) -> float:
        center_ids = np.asarray(center_ids, dtype=np.int32)
        pos_ids = np.asarray(pos_ids, dtype=np.int32)
        neg_ids = np.asarray(neg_ids, dtype=np.int32)

        v = self.V[center_ids]        # (B, D)
        u_pos = self.U[pos_ids]       # (B, D)
        u_neg = self.U[neg_ids]       # (B, K, D)

        pos_scores = np.sum(v * u_pos, axis=1)                       # (B,)
        neg_scores = np.sum(u_neg * v[:, None, :], axis=2)           # (B, K)

        # SGNS loss: -log sigmoida(pos) - sum logsigmoida(-neg)
        loss_pos = -np.log(sigmoid(pos_scores) + 1e-12)              # (B,)
        loss_neg = -np.log(sigmoid(-neg_scores) + 1e-12).sum(axis=1) # (B,)
        loss = (loss_pos + loss_neg).mean()

        self.cache = (center_ids, pos_ids, neg_ids, v, u_pos, u_neg, pos_scores, neg_scores)
        return float(loss)

    def backward(self):
        center_ids, pos_ids, neg_ids, v, u_pos, u_neg, pos_scores, neg_scores = self.cache
        B = center_ids.shape[0]

        # dL/dpos_scores = (σ(pos) - 1) / B
        g_pos = (sigmoid(pos_scores) - 1.0) / B                      # (B,)

        # dL/dneg_scores = σ(neg) / B   (for each negative)
        g_neg = sigmoid(neg_scores) / B                              # (B, K)

        # dv = g_pos*u_pos + sum_k g_neg_k*u_neg_k
        dv = g_pos[:, None] * u_pos + np.sum(g_neg[:, :, None] * u_neg, axis=1)  # (B, D)

        # du_pos = g_pos*v
        du_pos = g_pos[:, None] * v                                  # (B, D)

        # du_neg = g_neg * v
        du_neg = g_neg[:, :, None] * v[:, None, :]                   # (B, K, D)

        # scatter-add into dV and dU
        #np.add, indexes can be repated
        np.add.at(self.dV, center_ids, dv)
        np.add.at(self.dU, pos_ids, du_pos)

        # flatten negatives for add.at
        neg_flat = neg_ids.reshape(-1)                               # (B*K,)
        du_neg_flat = du_neg.reshape(-1, du_neg.shape[-1])           # (B*K, D)
        np.add.at(self.dU, neg_flat, du_neg_flat)

    def save(self, path: str):
        np.savez(
            path,
            V=self.V,
            U=self.U,
            lr=self.lr,
            word2id=self.word2id,
            id2word=self.id2word
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)

        self.V = data["V"]
        self.U = data["U"]
        self.lr = float(data["lr"])

        self.word2id = data["word2id"].item()
        self.id2word = data["id2word"].item()

        self.dV = np.zeros_like(self.V)
        self.dU = np.zeros_like(self.U)
        self.cache = None

    def step(self):
        self.V -= self.lr * self.dV
        self.U -= self.lr * self.dU