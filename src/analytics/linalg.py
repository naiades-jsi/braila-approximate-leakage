import numpy as np
import scipy.sparse

def normalize_rows_l1(X):
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    # normalize the rows
    X_abs = np.absolute(X)
    x_abs_rowsum = X_abs.sum(axis=1)

    nonzero_idxs = x_abs_rowsum >= 1e-5

    x_abs_rowsum_inv = np.reciprocal(x_abs_rowsum, where=nonzero_idxs)

    norm_mat = scipy.sparse.csr_matrix((x_abs_rowsum_inv, (range(n_rows), range(n_rows))), shape=(n_rows, n_rows))
    X = norm_mat.dot(X)

    return X

def col_multiply(X, col_coeff_vec):
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    print(X.shape)
    print(len(col_coeff_vec))

    diag_mat = scipy.sparse.csr_matrix((col_coeff_vec, (range(n_cols), range(n_cols))), shape=(n_cols, n_cols))
    return diag_mat.dot(X.T).T


def col_divide(X, col_coeff_vec):
    inv_coeff_vec = np.reciprocal(col_coeff_vec)
    return col_multiply(X, inv_coeff_vec)


def sub_all_comb(X1, X2):
    n_rows_x1 = X1.shape[0]
    n_rows_x2 = X2.shape[0]

    sp_data = []
    sp_rowidxs = []
    sp_colidxs = []
    for row1N in range(n_rows_x1):
        for row2N in range(n_rows_x2):
            sp_data.append(1)
            sp_colidxs.append(row1N)
            sp_rowidxs.append(row1N*n_rows_x2 + row2N)

    X1_transform_mat = scipy.sparse.csr_matrix((sp_data, (sp_rowidxs, sp_colidxs)), shape=(n_rows_x1*n_rows_x2, n_rows_x1))
    X1_dupl = X1_transform_mat.dot(X1)

    sp_data = []
    sp_rowidxs = []
    sp_colidxs = []
    for row1N in range(n_rows_x1):
        for row2N in range(n_rows_x2):
            sp_data.append(1)
            sp_colidxs.append(row2N)
            sp_rowidxs.append(row1N*n_rows_x2 + row2N)

    X2_transform_mat = scipy.sparse.csr_matrix((sp_data, (sp_rowidxs, sp_colidxs)), shape=(n_rows_x1*n_rows_x2, n_rows_x2))
    X2_dupl = X2_transform_mat.dot(X2)

    return X1_dupl - X2_dupl
