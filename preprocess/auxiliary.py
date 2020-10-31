import os

import numpy as np


def parse_icd9_range(range_: str) -> (str, str, int, int):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (1, 1, 1)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map) + 1, 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix


def generate_subclass_map(code_level_matrix: np.ndarray) -> list:
    code_num, level_num = code_level_matrix.shape
    max_level = np.max(code_level_matrix, axis=0)
    subclass_map = [[np.array(list(set((code_level_matrix[np.where(code_level_matrix[:, i] == l)[0]][:, i + 1]))),
                              dtype=int) - 1
                     for l in range(1, max_level[i] + 1)] for i in range(level_num - 1)]
    return subclass_map


def generate_code_code_adjacent(pids: np.ndarray, patient_admission: dict, admission_codes_encoded: dict, code_num: int) -> np.ndarray:
    print('generating code code adjacent matrix ...')
    n = code_num + 1
    result = np.zeros((n, n), dtype=float)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]:
            codes = admission_codes_encoded[admission['admission_id']]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    result[c_i, c_j] += 1
                    result[c_j, c_i] += 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    s = result.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = result / s
    result = result + np.eye(result.shape[0]) * 9
    # rowsum = result.sum(axis=-1)
    # degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    # result = result.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    result = result / result.sum(axis=-1, keepdims=True)
    return result.astype(np.float32)
