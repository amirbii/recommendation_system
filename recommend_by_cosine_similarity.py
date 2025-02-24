import numpy as np
import pandas as pd


def cosine_similarity_test(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


x = np.array([
    [1, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],  # 3
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0]
])
new_X = np.array([1, 1, 1, 1, 1, 0])

similarity = np.array([cosine_similarity_test(row, new_X) for row in x]).reshape(-1, 1)
print(similarity)

df = pd.DataFrame(x)
df['similarity'] = similarity
sort = df.sort_values(by='similarity', ascending=False)
filter_df = sort[sort['similarity'] < 0.99]
print(filter_df)

top = filter_df.head(1)
top_1 = top.drop(columns='similarity')
print("------------")
print(top_1)

top_2 = np.array(top_1).reshape(1, 6)
top_3 = np.squeeze(top_2)
print(top_3)
print(new_X)

out = top_3[new_X != top_3]
out_index = np.where(new_X != top_3)
print(f"recommend_value:{out}")
print(f"recommend_index:{out_index}")
