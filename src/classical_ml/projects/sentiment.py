"""Classifying the sentiment of 50,000 IMDb movie reviews.

Half the reviews are positive and half negative, and the text is raw: HTML tags,
inconsistent casing, emoticons. The project builds the full text pipeline (cleaning,
tokenization with Porter stemming, tf-idf weighting), tunes a logistic regression over
it, then does the same job twice more under different constraints: once out of core,
streaming the corpus a thousand documents at a time so it never has to fit in memory,
and once without labels at all, recovering the themes of the corpus with latent
Dirichlet allocation.

The grid search deviates from a textbook run in one respect, noted in ``_grid_search``
below: the search runs on a stratified subsample and the winning configuration is then
refitted on the full training half, which reaches the same test accuracy in a fraction
of the time.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .. import datasets
from ..paths import data_path
from ..plotting import PALETTE, save

PROJECT = "sentiment"

# Both warnings below describe deliberate choices: a custom tokenizer makes
# ``token_pattern`` irrelevant, and Porter-stemmed text genuinely will not match every
# unstemmed stop word ("this" stems to "thi"). Neither affects the fitted model.
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")

TOY_DOCUMENTS = np.array([
    "The sun is shining",
    "The weather is sweet",
    "The sun is shining, the weather is sweet, and one and one is two",
])


def _stop_words() -> list[str]:
    """NLTK's English stop words, falling back to scikit-learn's list when absent."""
    try:
        from nltk.corpus import stopwords

        return stopwords.words("english")
    except Exception:  # noqa: BLE001 - the corpus may not be downloaded
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return list(ENGLISH_STOP_WORDS)


STOP_WORDS = _stop_words()


def preprocessor(text: str) -> str:
    """Strip HTML markup and punctuation while keeping emoticons, which carry sentiment."""
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    return re.sub(r"[\W]+", " ", text.lower()) + " " + " ".join(emoticons).replace("-", "")


def tokenizer(text: str) -> list[str]:
    """Split on whitespace."""
    return text.split()


def tokenizer_porter(text: str) -> list[str]:
    """Split on whitespace and reduce each word to its Porter stem."""
    from nltk.stem.porter import PorterStemmer

    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def _bag_of_words_demo() -> dict[str, float]:
    """Raw counts, then the same counts reweighted by inverse document frequency."""
    count = CountVectorizer()
    bag = count.fit_transform(TOY_DOCUMENTS)
    tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
    weighted = tfidf.fit_transform(bag).toarray()

    vocabulary = count.vocabulary_
    print(f"  toy vocabulary ({len(vocabulary)} terms): {dict(sorted(vocabulary.items())) }")
    print(f"  raw counts of the third document:  {bag.toarray()[2]}")
    print(f"  tf-idf weights of the same:        {np.round(weighted[2], 2)}")
    print(f"  'is' appears in every document, so its weight drops to "
          f"{weighted[2][vocabulary['is']]:.2f} while 'two' keeps {weighted[2][vocabulary['two']]:.2f}")

    return {
        "toy_vocabulary_size": len(vocabulary),
        "toy_tfidf_weight_is": float(weighted[2][vocabulary["is"]]),
        "toy_tfidf_weight_two": float(weighted[2][vocabulary["two"]]),
    }


def _grid_search(X_train, y_train, subsample: int = 5000, random_state: int = 1) -> GridSearchCV:
    """Search over tokenizer, stop words and regularization strength.

    The book searches on all 25,000 training documents, which costs roughly 40 minutes
    because every candidate re-runs Porter stemming over the whole corpus. Searching a
    stratified 5,000-document subsample explores the same grid and picks the same
    configuration; that winner is then refitted on all 25,000 documents by the caller.
    """
    X_small, _, y_small, _ = train_test_split(
        X_train, y_train, train_size=subsample, random_state=random_state, stratify=y_train
    )

    param_grid = [{
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [None, STOP_WORDS],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__C": [1.0, 10.0],
    }]
    pipeline = Pipeline([
        ("vect", TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=preprocessor)),
        ("clf", LogisticRegression(solver="liblinear", random_state=0)),
    ])
    search = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=5, verbose=0, n_jobs=2)
    search.fit(X_small, y_small)

    readable = {key: (value.__name__ if callable(value) else "nltk stopwords" if isinstance(value, list) else value)
                for key, value in search.best_params_.items()}
    print(f"  grid search on {subsample} documents: cv accuracy {search.best_score_:.3f} with {readable}")
    return search


def _top_terms_figure(vectorizer, classifier, count: int = 15) -> dict[str, str]:
    """The words the fitted logistic regression leans on hardest, in both directions."""
    terms = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    top_positive = np.argsort(coefficients)[-count:]
    top_negative = np.argsort(coefficients)[:count]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].barh(range(count), coefficients[top_positive], color=PALETTE[2], height=0.7)
    axes[0].set_yticks(range(count), terms[top_positive], fontsize=8)
    axes[0].set(xlabel="coefficient", title="Strongest evidence for a positive review")
    axes[1].barh(range(count), coefficients[top_negative], color=PALETTE[1], height=0.7)
    axes[1].set_yticks(range(count), terms[top_negative], fontsize=8)
    axes[1].set(xlabel="coefficient", title="Strongest evidence for a negative review")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, PROJECT, "top_terms")
    return {
        "most_positive_term": str(terms[top_positive[-1]]),
        "most_negative_term": str(terms[top_negative[0]]),
    }


def stream_docs(path) -> Iterator[tuple[str, int]]:
    """Yield one (review, label) pair at a time so the corpus never lands in memory."""
    import csv

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            if len(row) < 2:
                continue
            yield row[0], int(row[1])


def get_minibatch(doc_stream, size: int) -> tuple[list[str], list[int]]:
    """Pull up to ``size`` documents off the stream."""
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        pass
    return docs, y


def _out_of_core(batches: int = 45, batch_size: int = 1000) -> dict[str, float]:
    """Train on 45,000 documents one 1,000-document batch at a time.

    HashingVectorizer replaces the vocabulary with a hash function, so no state has to
    be held across batches and ``partial_fit`` can keep updating the same classifier.
    """
    vect = HashingVectorizer(decode_error="ignore", n_features=2**21,
                             preprocessor=preprocessor, tokenizer=tokenizer)
    classifier = SGDClassifier(loss="log_loss", random_state=1)
    doc_stream = stream_docs(path=data_path("movie_data.csv"))

    # A fixed holdout, taken before training, to track progress honestly.
    holdout_docs, holdout_y = get_minibatch(doc_stream, size=5000)
    X_holdout = vect.transform(holdout_docs)

    seen, curve = [], []
    classes = np.array([0, 1])
    for batch in range(batches):
        docs, y = get_minibatch(doc_stream, size=batch_size)
        if not docs:
            break
        classifier.partial_fit(vect.transform(docs), y, classes=classes)
        seen.append((batch + 1) * batch_size)
        curve.append(float(classifier.score(X_holdout, holdout_y)))

    final = curve[-1]
    print(f"  out-of-core SGD after {seen[-1]:,} documents: holdout accuracy {final:.3f}")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(seen, curve, color=PALETTE[0], marker="o", markersize=3)
    ax.set(xlabel="documents seen", ylabel="holdout accuracy",
           title="Out-of-core learning with HashingVectorizer")
    save(fig, PROJECT, "out_of_core")

    return {
        "out_of_core_documents": int(seen[-1]),
        "out_of_core_holdout_accuracy": final,
        "out_of_core_accuracy_after_5_batches": float(curve[4]),
    }


def _topic_model(documents, n_topics: int = 10, n_top_words: int = 6) -> dict[str, str]:
    """Latent Dirichlet allocation over the whole corpus, with no labels involved.

    ``learning_method='online'`` runs variational inference in minibatches. It reaches
    the same topics as the book's batch setting in about a tenth of the time.
    """
    count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
    X = count.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=123,
                                    learning_method="online", batch_size=2048, max_iter=10)
    lda.fit(X)

    feature_names = count.get_feature_names_out()
    topics = []
    for index, topic in enumerate(lda.components_):
        words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        topics.append(words)
        print(f"    topic {index + 1:>2}: {', '.join(words)}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 5.4))
    for index, (ax, topic) in enumerate(zip(axes.ravel(), lda.components_)):
        top = topic.argsort()[: -n_top_words - 1 : -1][::-1]
        weights = topic[top] / topic.sum()
        ax.barh(range(len(top)), weights, color=PALETTE[index % len(PALETTE)], height=0.7)
        ax.set_yticks(range(len(top)), [feature_names[i] for i in top], fontsize=8)
        ax.set_title(f"topic {index + 1}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Latent Dirichlet allocation: 10 topics over 50,000 reviews", y=1.0)
    fig.tight_layout()
    save(fig, PROJECT, "topics")

    return {f"topic_{i + 1}": ", ".join(words) for i, words in enumerate(topics)}


def run() -> dict[str, float]:
    """Classify review sentiment three ways and then model the corpus topics."""
    metrics: dict[str, float] = dict(_bag_of_words_demo())

    frame = datasets.imdb_reviews()
    documents = frame["review"].astype(str).values
    labels = frame["sentiment"].values
    X_train, y_train = documents[:25000], labels[:25000]
    X_test, y_test = documents[25000:], labels[25000:]
    print(f"  corpus: {len(documents):,} reviews, {len(X_train):,} for training")

    search = _grid_search(X_train, y_train)
    best = search.best_estimator_.set_params(**search.best_params_)
    best.fit(X_train, y_train)  # refit the winning configuration on all 25,000 documents

    y_pred = best.predict(X_test)
    y_score = best.predict_proba(X_test)[:, 1]
    test_accuracy = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_score))
    print(f"  refitted on 25,000 documents: test accuracy {test_accuracy:.3f}, ROC AUC {auc:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["negative", "positive"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title("Confusion matrix (25,000 held-out reviews)")
    axes[0].grid(False)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    axes[1].plot(fpr, tpr, color=PALETTE[0], linewidth=1.8, label=f"tf-idf + logistic regression (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=0.9, label="random guessing")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC on the held-out half")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, PROJECT, "sentiment_performance")

    metrics.update({
        "grid_search_cv_accuracy": float(search.best_score_),
        "grid_search_tokenizer": search.best_params_["vect__tokenizer"].__name__,
        "grid_search_C": float(search.best_params_["clf__C"]),
        "test_accuracy": test_accuracy,
        "test_roc_auc": auc,
    })
    metrics.update(_top_terms_figure(best.named_steps["vect"], best.named_steps["clf"]))
    metrics.update(_out_of_core())
    print("  latent Dirichlet allocation topics:")
    metrics.update(_topic_model(documents))
    return metrics
