import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def get_top_n_similar_words(
    model,
    word: str,
    word2id: dict,
    id2word: dict,
    n: int = 10,
    use_central: bool = True
):

    if word not in word2id:
        raise ValueError(f"Word '{word}' not in vocabulary")

    word_id = word2id[word]

    if use_central:
        embeddings = model.V
    else:
        embeddings = model.U

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms

    query_vec = normalized[word_id]
    similarities = normalized @ query_vec


    similarities[word_id] = -np.inf

    top_ids = np.argsort(similarities)[-n:][::-1]

    results = [(id2word[i], float(similarities[i])) for i in top_ids]

    return results


def plot_top_words_map(model, popular_words: int, word_counts: dict):
    selected_embeddings = []
    selected_words = []

    for word, _ in word_counts.most_common(popular_words):
        id_ = model.word2id[word]
        selected_embeddings.append(model.V[id_])
        selected_words.append(word)

    selected_embeddings = np.array(selected_embeddings)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(selected_embeddings)

    df = pd.DataFrame({
        "word": selected_words,
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="word",
        hover_name="word"
    )

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(width=1000, height=800)

    fig.show()


def plot_example_cluster(model):
    groups = {
        "days": [
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ],
        "months": [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"
        ],
        "directions": [
            "north", "south", "east", "west"
        ],
        "colors": [
            "red", "blue", "green", "yellow", "black", "white"
        ],
        "animals": [
            "dog", "cat", "horse", "cow", "sheep", "goat"
        ]
    }

    selected_embeddings = []
    selected_words = []
    true_groups = []

    for group, words in groups.items():
        for w in words:
            if w in model.word2id:
                selected_embeddings.append(model.V[model.word2id[w]])
                selected_words.append(w)
                true_groups.append(group)

    if len(selected_embeddings) < 2:
        print("Not enough words for visualization!")
        return

    selected_embeddings = np.array(selected_embeddings)
    print("Shape:", selected_embeddings.shape)

    n_groups = len(groups)
    n_clusters = min(n_groups, len(selected_embeddings))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(selected_embeddings)

    perplexity = min(5, len(selected_embeddings) - 1)
    reduced = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity
    ).fit_transform(selected_embeddings)

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "word": selected_words,
        "true_group": true_groups,
        "cluster": clusters.astype(str)
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        text="word",
        hover_data=["true_group"],
        title="KMeans clustering of Word2Vec embeddings"
    )

    fig.update_traces(textposition="top center")
    fig.show()

def plot_capital_relationships(model):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    V = model.V

    pairs = [
        ("germany", "berlin"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("netherlands", "amsterdam"),
        ("austria", "vienna"),
        ("czech republic", "prague"),
        ("denmark", "copenhagen"),
        ("finland", "helsinki"),
    ]

    # pairs which are in dictionary
    pairs = [(a, b) for a, b in pairs if a in model.word2id and b in model.word2id]

    all_words = [w for pair in pairs for w in pair]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    # PCA в 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    country_coords = coords[0::2]
    capital_coords = coords[1::2]

    plt.figure(figsize=(12, 7))

    plt.scatter(country_coords[:, 0], country_coords[:, 1], s=20)
    plt.scatter(capital_coords[:, 0], capital_coords[:, 1], s=20)

    for i, (country, capital) in enumerate(pairs):
        x1, y1 = country_coords[i]
        x2, y2 = capital_coords[i]

        plt.plot([x1, x2], [y1, y2], '--', alpha=0.5, linewidth=1)

        plt.text(x1 - 0.03, y1, country, fontsize=10, ha='right', va='center')
        plt.text(x2 + 0.03, y2, capital, fontsize=10, ha='left', va='center')

    plt.title("Country and Capital Vectors Projected by PCA")
    plt.xlabel("Word2Vec / PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_comparative_relations(model):
    V = model.V

    triples = [
        ("small", "smaller", "smallest"),
        ("big", "bigger", "biggest"),
        ("strong", "stronger", "strongest"),
        ("fast", "faster", "fastest"),
        ("long", "longer", "longest"),
        ("high", "higher", "highest"),
        ("young", "younger", "youngest"),
    ]

    triples = [
        (a, b, c) for a, b, c in triples
        if a in model.word2id and b in model.word2id and c in model.word2id
    ]

    print(len(triples))
    if len(triples) == 0:
        raise ValueError("No one triples in the dictionary!")

    all_words = [w for triple in triples for w in triple]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    base_coords = coords[0::3]
    comp_coords = coords[1::3]
    sup_coords = coords[2::3]

    plt.figure(figsize=(12, 8))

    plt.scatter(base_coords[:, 0], base_coords[:, 1], s=25, label="base")
    plt.scatter(comp_coords[:, 0], comp_coords[:, 1], s=25, label="comparative")
    plt.scatter(sup_coords[:, 0], sup_coords[:, 1], s=25, label="superlative")

    for i, (base, comp, sup) in enumerate(triples):
        x1, y1 = base_coords[i]
        x2, y2 = comp_coords[i]
        x3, y3 = sup_coords[i]

        # base -> comparative
        plt.plot([x1, x2], [y1, y2], '--', alpha=0.6, linewidth=1)

        # comparative -> superlative
        plt.plot([x2, x3], [y2, y3], '--', alpha=0.6, linewidth=1)

        plt.text(x1 - 0.01, y1, base, fontsize=10, ha='right', va='center')
        plt.text(x2 + 0.01, y2, comp, fontsize=10, ha='left', va='center')
        plt.text(x3 + 0.01, y3, sup, fontsize=10, ha='left', va='center')

    plt.title("Adjective → Comparative → Superlative projected by PCA")
    plt.xlabel("Word2Vec / PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_plural_relations(model):
    plural_pairs = [
        ("cat", "cats"),
        ("dog", "dogs"),
        ("car", "cars"),
        ("tree", "trees"),
        ("house", "houses"),
        ("book", "books"),
        ("river", "rivers"),
        ("city", "cities"),
        ("mouse", "mice")
    ]

    sing = []
    plur = []
    labels_s = []
    labels_p = []

    for s, p in plural_pairs:
        if s in model.word2id and p in model.word2id:
            sing.append(model.V[model.word2id[s]])
            plur.append(model.V[model.word2id[p]])
            labels_s.append(s)
            labels_p.append(p)

    X = np.vstack(sing + plur)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    n = len(sing)

    sing_pca = X2[:n]
    plur_pca = X2[n:]

    plt.figure(figsize=(12, 8))

    plt.scatter(sing_pca[:, 0], sing_pca[:, 1], label="singular")
    plt.scatter(plur_pca[:, 0], plur_pca[:, 1], label="plural")

    for i in range(n):
        plt.plot(
            [sing_pca[i, 0], plur_pca[i, 0]],
            [sing_pca[i, 1], plur_pca[i, 1]],
            linestyle="--"
        )

        plt.text(sing_pca[i, 0], sing_pca[i, 1], labels_s[i])
        plt.text(plur_pca[i, 0], plur_pca[i, 1], labels_p[i])

    plt.legend()
    plt.title("Singular → Plural relation in embeddings")
    plt.show()