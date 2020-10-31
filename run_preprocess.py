import os
import _pickle as pickle

from preprocess.parse_csv import Mimic3Parser, EICUParser
from preprocess.encode import encode_code
from preprocess.build_dataset import split_patients
from preprocess.build_dataset import build_code_xy, build_hierarchical_y, build_heart_failure_y
from preprocess.auxiliary import generate_code_levels, generate_subclass_map, generate_code_code_adjacent


if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num': 6000,
            'test_num': 1000
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'test_num': 1000
        }
    }
    data_path = 'data'
    dataset = 'eicu'  # 'mimic3' or 'eicu'
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parser = conf[dataset]['parser'](raw_path)
    patient_admission, admission_codes = parser.parse()

    print('There are %d valid patients' % len(patient_admission))

    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    max_code_num_in_a_visit = max([len(codes) for codes in admission_codes.values()])
    print('max admission num: %d, max code num in an admission: %d' % (max_admission_num, max_code_num_in_a_visit))

    admission_codes_encoded, code_map, code_map_pretrain = encode_code(patient_admission, admission_codes)

    code_num = len(code_map)
    code_num_pretrain = len(code_map_pretrain)
    print('There are %d pretrain codes, %d codes in multiple visits' % (code_num_pretrain, code_num))

    pretrain_pids, train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        test_num=conf[dataset]['test_num']
    )
    print('There are %d pretrain, %d train, %d valid, %d test samples' %
          (len(pretrain_pids), len(train_pids), len(valid_pids), len(test_pids)))

    code_levels_pretrain = generate_code_levels(data_path, code_map_pretrain)
    code_levels = code_levels_pretrain[:(code_num + 1)]
    subclass_maps_pretrain = generate_subclass_map(code_level_matrix=code_levels_pretrain)
    subclass_maps = generate_subclass_map(code_level_matrix=code_levels)
    code_code_adj = generate_code_code_adjacent(pids=pretrain_pids, patient_admission=patient_admission,
                                                admission_codes_encoded=admission_codes_encoded,
                                                code_num=code_num_pretrain)

    pretrain_codes_x, pretrain_codes_y, pretrain_visit_lens = build_code_xy(pretrain_pids, patient_admission,
                                                                            admission_codes_encoded, max_admission_num,
                                                                            code_num_pretrain, max_code_num_in_a_visit,
                                                                            pretrain=True)

    train_codes_x, train_codes_y, train_visit_lens = build_code_xy(train_pids, patient_admission,
                                                                   admission_codes_encoded, max_admission_num,
                                                                   code_num, max_code_num_in_a_visit)
    valid_codes_x, valid_codes_y, valid_visit_lens = build_code_xy(valid_pids, patient_admission,
                                                                   admission_codes_encoded, max_admission_num,
                                                                   code_num, max_code_num_in_a_visit)
    test_codes_x, test_codes_y, test_visit_lens = build_code_xy(test_pids, patient_admission, admission_codes_encoded,
                                                                max_admission_num, code_num,
                                                                max_code_num_in_a_visit)

    pretrain_y_h = build_hierarchical_y(code_levels_pretrain, pretrain_codes_y)
    train_y_h = build_hierarchical_y(code_levels, train_codes_y)
    valid_y_h = build_hierarchical_y(code_levels, valid_codes_y)
    test_y_h = build_hierarchical_y(code_levels, test_codes_y)

    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    pretrain_codes_data = (pretrain_codes_x, pretrain_codes_y, pretrain_y_h, pretrain_visit_lens)
    train_codes_data = (train_codes_x, train_codes_y, train_y_h, train_visit_lens)
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_y_h, valid_visit_lens)
    test_codes_data = (test_codes_x, test_codes_y, test_y_h, test_visit_lens)

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump({
        'code_map': code_map,
        'code_map_pretrain': code_map_pretrain
    }, open(os.path.join(encoded_path, 'code_maps.pkl'), 'wb'))
    pickle.dump({
        'pretrain_pids': pretrain_pids,
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    pickle.dump(pretrain_codes_data, open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_hf_y': train_hf_y,
        'valid_hf_y': valid_hf_y,
        'test_hf_y': test_hf_y
    }, open(os.path.join(standard_path, 'heart_failure.pkl'), 'wb'))
    pickle.dump({
        'code_levels': code_levels,
        'code_levels_pretrain': code_levels_pretrain,
        'subclass_maps': subclass_maps,
        'subclass_maps_pretrain': subclass_maps_pretrain,
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))
