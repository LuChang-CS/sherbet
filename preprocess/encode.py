import copy
from collections import OrderedDict


def encode_code(patient_admission: dict, admission_codes: dict) -> (dict, dict, dict):
    print('encoding code ...')
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        if len(admissions) <= 1:
            continue
        for admission in admissions:
            codes = admission_codes[admission['admission_id']]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map) + 1
    code_map_pretrain = copy.deepcopy(code_map)
    for pid, admissions in patient_admission.items():
        if len(admissions) > 1:
            continue
        for admission in admissions:
            codes = admission_codes[admission['admission_id']]
            for code in codes:
                if code not in code_map_pretrain:
                    code_map_pretrain[code] = len(code_map_pretrain) + 1

    admission_codes_encoded = {
        admission_id: [code_map_pretrain[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map, code_map_pretrain
