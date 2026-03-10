import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA


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

    fig.write_html("top_words_map.html")
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
    fig.write_html("clusters_map.html")
    fig.show()


def plot_capital_relationships(model):
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

    # Leave only pairs that exist in the vocabulary
    pairs = [(country, capital) for country, capital in pairs
             if country in model.word2id and capital in model.word2id]

    if not pairs:
        raise ValueError("No country-capital pairs were found in the model vocabulary.")

    all_words = [w for pair in pairs for w in pair]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    country_coords = coords[0::2]
    capital_coords = coords[1::2]

    # DataFrames for convenient plotting
    df_countries = pd.DataFrame({
        "word": [country for country, _ in pairs],
        "x": country_coords[:, 0],
        "y": country_coords[:, 1],
        "type": "country",
    })

    df_capitals = pd.DataFrame({
        "word": [capital for _, capital in pairs],
        "x": capital_coords[:, 0],
        "y": capital_coords[:, 1],
        "type": "capital",
    })

    fig = go.Figure()

    # Countries
    fig.add_trace(go.Scatter(
        x=df_countries["x"],
        y=df_countries["y"],
        mode="markers+text",
        text=df_countries["word"],
        textposition="middle left",
        name="Countries",
        hovertemplate="<b>%{text}</b><br>Type: country<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        marker=dict(size=10)
    ))

    # Capitals
    fig.add_trace(go.Scatter(
        x=df_capitals["x"],
        y=df_capitals["y"],
        mode="markers+text",
        text=df_capitals["word"],
        textposition="middle right",
        name="Capitals",
        hovertemplate="<b>%{text}</b><br>Type: capital<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        marker=dict(size=10)
    ))

    # Dashed lines between country and capital
    for i, (country, capital) in enumerate(pairs):
        x1, y1 = country_coords[i]
        x2, y2 = capital_coords[i]

        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="Country and Capital Vectors Projected by PCA",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=800,
        height=600,
        template="plotly_white"
    )

    fig.show()
    fig.write_html("capital_relationships.html")


def plot_comparative_relations(
    model
):
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

    if len(triples) == 0:
        raise ValueError("No triples in the dictionary!")

    all_words = [w for triple in triples for w in triple]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    base_coords = coords[0::3]
    comp_coords = coords[1::3]
    sup_coords = coords[2::3]

    df_base = pd.DataFrame({
        "word": [a for a, _, _ in triples],
        "x": base_coords[:, 0],
        "y": base_coords[:, 1],
        "type": "base"
    })

    df_comp = pd.DataFrame({
        "word": [b for _, b, _ in triples],
        "x": comp_coords[:, 0],
        "y": comp_coords[:, 1],
        "type": "comparative"
    })

    df_sup = pd.DataFrame({
        "word": [c for _, _, c in triples],
        "x": sup_coords[:, 0],
        "y": sup_coords[:, 1],
        "type": "superlative"
    })

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=df_base["x"],
        y=df_base["y"],
        mode="markers+text",
        text=df_base["word"],
        textposition="bottom center",
        name="base",
        marker=dict(size=10),
        hovertemplate="<b>%{text}</b><br>Type: base<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_comp["x"],
        y=df_comp["y"],
        mode="markers+text",
        text=df_comp["word"],
        textposition="bottom center",
        name="comparative",
        marker=dict(size=10),
        hovertemplate="<b>%{text}</b><br>Type: comparative<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_sup["x"],
        y=df_sup["y"],
        mode="markers+text",
        text=df_sup["word"],
        textposition="bottom center",
        name="superlative",
        marker=dict(size=10),
        hovertemplate="<b>%{text}</b><br>Type: superlative<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))

    for i, (base, comp, sup) in enumerate(triples):
        x1, y1 = base_coords[i]
        x2, y2 = comp_coords[i]
        x3, y3 = sup_coords[i]

        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=[x2, x3],
            y=[y2, y3],
            mode="lines",
            line=dict(dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="Adjective → Comparative → Superlative projected by PCA",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=970,
        height=800,
        template="plotly_white"
    )

    fig.show()
    fig.write_html("comparative_relations.html")

def plot_plural_relations(
    model):

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

    if len(sing) == 0:
        raise ValueError("No singular/plural pairs found in the dictionary!")

    X = np.vstack(sing + plur)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    n = len(sing)

    sing_pca = X2[:n]
    plur_pca = X2[n:]

    df_sing = pd.DataFrame({
        "word": labels_s,
        "x": sing_pca[:, 0],
        "y": sing_pca[:, 1],
        "type": "singular"
    })

    df_plur = pd.DataFrame({
        "word": labels_p,
        "x": plur_pca[:, 0],
        "y": plur_pca[:, 1],
        "type": "plural"
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sing["x"],
        y=df_sing["y"],
        mode="markers+text",
        text=df_sing["word"],
        textposition="middle left",
        name="singular",
        marker=dict(size=10),
        hovertemplate="<b>%{text}</b><br>Type: singular<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_plur["x"],
        y=df_plur["y"],
        mode="markers+text",
        text=df_plur["word"],
        textposition="middle right",
        name="plural",
        marker=dict(size=10),
        hovertemplate="<b>%{text}</b><br>Type: plural<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
    ))

    for i in range(n):
        fig.add_trace(go.Scatter(
            x=[sing_pca[i, 0], plur_pca[i, 0]],
            y=[sing_pca[i, 1], plur_pca[i, 1]],
            mode="lines",
            line=dict(dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="Singular → Plural relation in embeddings",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=1100,
        height=800,
        template="plotly_white"
    )

    fig.show()
    fig.write_html("plural_relations.html")