from typing import List

def all_rows_have_same_length(mat: List[List[any]]):
    if not row_has_data(mat):
        raise ValueError('Empty matrix is not allowed')

    prev = len(mat[0])
    for i in (range(1, len(mat))):
        if prev != len(mat[i]):
            raise ValueError('All rows must have the same length in the data')

def row_has_data(row: List[any]) -> bool:
    return len(row) != 0