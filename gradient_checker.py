import numpy as np

def grad_check_used_only(model, center_ids, pos_ids, neg_ids, eps=1e-5,
                         rtol=1e-5):
    model.zero_grad()
    model.forward(center_ids, pos_ids, neg_ids)
    model.backward()

    used_V_rows = np.unique(center_ids)
    used_U_rows = np.unique(np.concatenate([pos_ids.reshape(-1), neg_ids.reshape(-1)]))

    print("Checking V...")
    for i in used_V_rows:
        for j in range(model.V.shape[1]):
            old = model.V[i, j]

            model.V[i, j] = old + eps
            lp = model.forward(center_ids, pos_ids, neg_ids)

            model.V[i, j] = old - eps
            lm = model.forward(center_ids, pos_ids, neg_ids)

            model.V[i, j] = old

            gnum = (lp - lm) / (2 * eps)
            gan = model.dV[i, j]

            rel = abs(gnum - gan) / max(1e-8, abs(gnum), abs(gan))

            print(f"V[{i},{j}] rel={rel:.3e}")
            if rel > rtol:
                print("NOT PASSED")
                return

    print("Checking U...")
    for i in used_U_rows:
        for j in range(model.U.shape[1]):
            old = model.U[i, j]

            model.U[i, j] = old + eps
            lp = model.forward(center_ids, pos_ids, neg_ids)

            model.U[i, j] = old - eps
            lm = model.forward(center_ids, pos_ids, neg_ids)

            model.U[i, j] = old

            gnum = (lp - lm) / (2 * eps)
            gan = model.dU[i, j]
            rel = abs(gnum - gan) / max(1e-8, abs(gnum), abs(gan))

            print(f"U[{i},{j}] rel={rel:.3e}")
            if rel > rtol:
                print("NOT PASSED")
                return

    print("Gradient check passed")