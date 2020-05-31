import numpy as np


def arr_print(arr):
    representation = ""
    rows, cols = arr.shape
    for row in range(rows):
        for col in range(cols):
            if arr[row, col] == np.iinfo(np.int64).min:
                value = 'x'.rjust(10)
            else:
                value = round(arr[row, col], 5)
            representation += f"{value:10}"
            representation += " "
        representation += "\n"
    return representation


def prep_maze(arr):
    empty_cells = arr == 0
    arr[empty_cells] = np.iinfo(np.int64).min
    return arr


def prep_val_est_arr(arr):
    value_estimates = np.zeros_like(arr, dtype=np.float)
    empty_cells = arr == np.iinfo(np.int64).min
    value_estimates[empty_cells] = np.iinfo(np.int64).min
    return value_estimates


def prep_results(results):
    results = list(map(lambda x: x.tolist(), results))
    for result in results:
        for j, row in enumerate(result):
            result[j] = list(map(lambda x: x if x != np.iinfo(np.int64).min else 'x', row))
    return results
