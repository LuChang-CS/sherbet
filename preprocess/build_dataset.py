import itertools

import numpy as np


def split_patients(patient_admission: dict,
                   admission_codes: dict,
                   code_map: dict,
                   train_num,
                   test_num,
                   seed=6669) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    print('splitting pretrain, train, valid, and test pids ...')
    single = []
    multiple = []
    for pid, admissions in patient_admission.items():
        if len(admissions) > 1:
            multiple.append(pid)
        else:
            single.append(pid)
    print('There are %d single admission patients, %d multiple admission patients' % (len(single), len(multiple)))
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid in multiple:
            admissions = patient_admission[pid]
            for admission in admissions:
                codes = admission_codes[admission['admission_id']]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')

    pid_max_admission_num, _ = max([(pid, len(patient_admission[pid])) for pid in multiple], key=lambda x: x[1])
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(multiple).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(multiple) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    pretrain_pids = np.concatenate([np.array(single), train_pids])
    return pretrain_pids, train_pids, valid_pids, test_pids


def build_code_xy(pids: np.ndarray,
                  patient_admission: dict,
                  admission_codes_encoded: dict,
                  max_admission_num: int,
                  code_num: int,
                  max_code_num_in_a_visit: int,
                  pretrain=False) -> (np.ndarray, np.ndarray, np.ndarray):
    print('building pretrain/train/valid/test codes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        end_pos = None if pretrain else -1
        for k, admission in enumerate(admissions[:end_pos]):
            codes = admission_codes_encoded[admission['admission_id']]
            x[i][k][:len(codes)] = codes
        if pretrain:
            codes = set(itertools.chain.from_iterable([admission_codes_encoded[admission['admission_id']]
                                                       for admission in admissions]))
            codes = np.array(list(codes)) - 1
        else:
            codes = np.array(admission_codes_encoded[admissions[-1]['admission_id']]) - 1
        y[i][codes] = 1
        lens[i] = len(admissions) if pretrain else len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_hierarchical_y(code_levels, y):
    print('building pretrain/train/valid/test hierarchical labels ...')
    subclass_dims = np.max(code_levels, axis=0)
    n = len(y)
    y_trues = [np.zeros((n, dim), dtype=int) for dim in subclass_dims]
    for i, codes_hot in enumerate(y):
        print('\r\t%d / %d' % (i + 1, n), end='')
        codes = np.where(codes_hot == 1)[0] + 1
        for code in codes:
            levels = code_levels[code]
            for l, level in enumerate(levels):
                y_trues[l][i][level - 1] = 1
    print('\r\t%d / %d' % (n, n))
    return y_trues


def build_heart_failure_y(hf_prefix: str, codes_y: np.ndarray, code_map: dict) -> np.ndarray:
    print('building train/valid/test heart failure labels ...')
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map), ), dtype=int)
    hfs[hf_list - 1] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
