"""Loads the public labeled dataset. Label 1 = prompt injection, 0 = harmless."""

DATASET = "deepset/prompt-injections"  # check its license on the dataset page


def load_data():
    """Returns (train_texts, train_labels, test_texts, test_labels)."""
    from datasets import load_dataset  # imported here so the tests do not need the download

    ds = load_dataset(DATASET)
    train, test = ds["train"], ds["test"]
    return (list(train["text"]), list(train["label"]),
            list(test["text"]), list(test["label"]))
