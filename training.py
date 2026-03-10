from tqdm import tqdm
import numpy as np
import random

def dataloader(data, batch_size, shuffle=True):
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [data[i] for i in batch_indices]

        central_ids = [x[0] for x in batch]
        pos_context_ids = [x[1] for x in batch]

        yield central_ids, pos_context_ids

def sample_negatives_for_batch(
    central_ids: np.ndarray,
    pos_ids: np.ndarray,
    k: int,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    B = len(central_ids)
    vocab_size = len(probs)

    neg_ids = np.zeros((B, k), dtype=np.int32)

    for i in range(B):
        forbidden = {int(central_ids[i]), int(pos_ids[i])}
        negs = []

        while len(negs) < k:
            draw = rng.choice(vocab_size, size=(k - len(negs)) * 3,
                              replace=True, p=probs)
            for w in draw:
                w = int(w)
                if w in forbidden:
                    continue
                negs.append(w)
                if len(negs) == k:
                    break

        neg_ids[i] = negs

    return neg_ids

def train(model, data, num_epochs: int, k: int = 5, batch_size: int = 32,
          probs=None, rng=None, dataset_name: str = "text8"):

    total_words = num_epochs * len(data) #for lr
    words_processed = 0
    initial_lr = 0.05

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        batches = 0

        for central_ids, pos_ids in dataloader(data, batch_size=batch_size):
            neg_ids = sample_negatives_for_batch(central_ids, pos_ids, k=k, probs=probs, rng=rng)
            lr = max(initial_lr * (1 - words_processed / total_words),
                     initial_lr * 0.0001)
            model.lr = lr
            model.zero_grad()
            last_loss = model.forward(central_ids, pos_ids, neg_ids)
            model.backward()
            model.step()

            total_loss += last_loss
            batches += 1
            words_processed += batch_size

        print(f"epoch {epoch+1}: loss={total_loss/batches:.4f}")
        model.save(f"{dataset_name}_w2v_n{len(data)}_v{model.V.shape[0]}_d{model.V.shape[1]}_e{epoch+1}.npz")