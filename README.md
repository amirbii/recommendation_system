# Recommend by Cosine Similarity

This project is a simple recommender system based on **cosine similarity**, utilizing **NumPy** and **Pandas** for data processing.

## Features

- Computes cosine similarity between a new vector and a set of existing vectors.
- Sorts and filters results based on similarity scores.
- Suggests features that are missing in the new vector but present in the closest match.

## Prerequisites

To run this project, make sure you have Python installed along with the following dependencies:

```bash
pip install numpy pandas
```

## How to Run

Execute the `recommend_by_cosine_similarity.py` file:

```bash
python recommend_by_cosine_similarity.py
```

## Sample Output

```bash
[[0.866] [0.775] [0.99] [0.91] [0.82] [0.63] [0.99]]
------------
Closest vector: [0 1 1 1 1 0]
Recommended features: [0]
Recommended feature indices: (array([0]),)
```



