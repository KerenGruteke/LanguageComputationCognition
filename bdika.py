import numpy as np
import copy


def get_avg_vectors_per_category(vectors):
    vectors = copy.deepcopy(vectors)
    avg_vectors_per_category = {}
    for category_name in ["A"]:
        avg_vectors_per_category[category_name] = 0

    for idx, vec in enumerate(vectors):
        name = "A"
        if avg_vectors_per_category[name] == 0:
            avg_vectors_per_category[name] = [vec, 1]
        else:
            avg_vectors_per_category[name][0] += vec
            avg_vectors_per_category[name][1] += 1

    avg_vectors_per_category_list = []
    for category_values in avg_vectors_per_category.values():
        avg = [x / category_values[1] for x in category_values[0]]
        avg_vectors_per_category_list.append(avg)

    return avg_vectors_per_category_list


if __name__ == "__main__":
    a = np.array([1, 1, 1])
    b = np.array([3, 0, 3])
    c = np.array([2, 0, 5])
    vectors = [a, b, c]
    print(get_avg_vectors_per_category(vectors=vectors))
    print(a)
    print(b)
    print(c)
