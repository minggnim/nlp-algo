import pickle


def save_augmented_corpus(augmented_data: list, file: str = '../data/augmented_corpus.pkl') -> None:
    with open(file, 'wb') as f:
        pickle.dump(augmented_data, f)