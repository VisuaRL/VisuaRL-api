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


def prep_arrows(results):
    # results - list of np.ndarrays (value arrs)
    def process_single_arr(arr):
        rows, cols = arr.shape
        output = np.zeros_like(arr, dtype=np.int64)
        for row in range(rows):
            for col in range(cols):
                if arr[row, col] == np.iinfo(np.int64).min:
                    continue
                out = ['0'] * 4
                max_val = np.iinfo(np.int64).min
                # Get max value
                if not row - 1 < 0:
                    max_val = max_val if arr[row - 1, col] < max_val else arr[row - 1, col]
                if not row + 1 >= rows:
                    max_val = max_val if arr[row + 1, col] < max_val else arr[row + 1, col]
                if not col - 1 < 0:
                    max_val = max_val if arr[row, col - 1] < max_val else arr[row, col - 1]
                if not col + 1 >= cols:
                    max_val = max_val if arr[row, col + 1] < max_val else arr[row, col + 1]
                # Compute arrows
                if not row - 1 < 0:
                    if max_val == arr[row - 1, col]:
                        out[0] = '1'
                if not row + 1 >= rows:
                    if max_val == arr[row + 1, col]:
                        out[1] = '1'
                if not col - 1 < 0:
                    if max_val == arr[row, col - 1]:
                        out[2] = '1'
                if not col + 1 >= cols:
                    if max_val == arr[row, col + 1]:
                        out[3] = '1'
                output[row, col] = int("".join(out), 2)
        return output
    return list(map(process_single_arr, results))
