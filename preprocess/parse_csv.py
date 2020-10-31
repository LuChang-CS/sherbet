import os
from datetime import datetime
from collections import OrderedDict

import pandas as pd
import numpy as np


class EHRParser:
    def __init__(self, path):
        self.path = path
        self.admission_csv = ''
        self.diagnosis_csv = ''
        self.admission_cols = {
            'pid': None,
            'adm_id': None,
            'adm_time': None
        }
        self.diagnosis_cols = {
            'pid': None,
            'adm_id': None,
            'cid': None
        }
        self.skip_pid_check = False
        self.patient_admission = None
        self.admission_codes = None

        self.admission_col_converter = self.set_admission_col()
        self.diagnosis_col_converter = self.set_diagnosis_col()

    def set_admission_col(self) -> dict:
        raise NotImplementedError

    def set_diagnosis_col(self) -> dict:
        raise NotImplementedError

    def to_standard_icd9(self, code: str):
        raise NotImplementedError

    def parse_admission(self):
        print('parsing the csv file of admission ...')
        admission_path = os.path.join(self.path, self.admission_csv)
        admissions = pd.read_csv(
            admission_path,
            usecols=list(self.admission_cols.values()),
            converters=self.admission_col_converter
        )
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid = row[self.admission_cols['pid']]
            admission_id = row[self.admission_cols['adm_id']]
            admission_time = row[self.admission_cols['adm_time']]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({
                'admission_id': admission_id,
                'admission_time': admission_time
            })
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            patient_admission[pid] = sorted(admissions, key=lambda admission: admission['admission_time'])

        self.patient_admission = patient_admission

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        diagnoses_path = os.path.join(self.path, self.diagnosis_csv)
        diagnoses = pd.read_csv(
            diagnoses_path,
            usecols=list(self.diagnosis_cols.values()),
            converters=self.diagnosis_col_converter
        )

        admission_codes = OrderedDict()
        for i, row in diagnoses.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(diagnoses)), end='')
            pid = row[self.diagnosis_cols['pid']]
            if self.skip_pid_check or pid in self.patient_admission:
                admission_id = row[self.diagnosis_cols['adm_id']]
                code = row[self.diagnosis_cols['cid']]
                code = self.to_standard_icd9(code)
                if code == '':
                    continue
                if admission_id not in admission_codes:
                    codes = []
                    admission_codes[admission_id] = codes
                else:
                    codes = admission_codes[admission_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(diagnoses), len(diagnoses)))

        self.admission_codes = admission_codes

        self.after_parse_diagnosis()

    def after_parse_diagnosis(self):
        pass

    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                if admission['admission_id'] not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                if admission['admission_id'] in self.admission_codes:
                    del self.admission_codes[admission['admission_id']]
                else:
                    # print('\tpatient %d have an admission %d without diagnosis' % (pid, admission['admission_id']))
                    pass
            del self.patient_admission[pid]

    def parse(self):
        self.parse_admission()
        self.parse_diagnoses()
        self.calibrate_patient_by_admission()
        return self.patient_admission, self.admission_codes


class Mimic3Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.admission_csv = 'ADMISSIONS.csv'
        self.diagnosis_csv = 'DIAGNOSES_ICD.csv'

    def set_admission_col(self) -> dict:
        self.admission_cols['pid'] = 'SUBJECT_ID'
        self.admission_cols['adm_id'] = 'HADM_ID'
        self.admission_cols['adm_time'] = 'ADMITTIME'
        converter = {
            'SUBJECT_ID': np.int,
            'HADM_ID': np.int,
            'ADMITTIME': lambda cell: datetime.strptime(np.str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return converter

    def set_diagnosis_col(self) -> dict:
        self.diagnosis_cols['pid'] = 'SUBJECT_ID'
        self.diagnosis_cols['adm_id'] = 'HADM_ID'
        self.diagnosis_cols['cid'] = 'ICD9_CODE'
        converter = {'SUBJECT_ID': np.int, 'HADM_ID': np.int, 'ICD9_CODE': np.str}
        return converter

    def to_standard_icd9(self, code: str):
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


class EICUParser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.admission_csv = 'patient.csv'
        self.diagnosis_csv = 'diagnosis.csv'
        self.skip_pid_check = True

    def set_admission_col(self) -> dict:
        self.admission_cols['pid'] = 'patienthealthsystemstayid'
        self.admission_cols['adm_id'] = 'patientunitstayid'
        self.admission_cols['adm_time'] = 'hospitaladmitoffset'
        converter = {
            'patienthealthsystemstayid': np.int,
            'patientunitstayid': np.int,
            'hospitaladmitoffset': lambda cell: -np.int(cell)
        }
        return converter

    def set_diagnosis_col(self) -> dict:
        self.diagnosis_cols['pid'] = 'diagnosisid'
        self.diagnosis_cols['adm_id'] = 'patientunitstayid'
        self.diagnosis_cols['cid'] = 'icd9code'
        converter = {'diagnosisid': np.int, 'patientunitstayid': np.int, 'icd9code': np.str}
        return converter

    def to_standard_icd9(self, code: str):
        if code == '':
            return code
        code = code.split(',')[0]
        c = code[0].lower()
        dot = code.find('.')
        if dot == -1:
            dot = None
        if not c.isalpha():
            prefix = code[:dot]
            if len(prefix) < 3:
                code = ('%03d' % int(prefix)) + code[dot:]
            return code
        if c == 'e':
            prefix = code[1:dot]
            if len(prefix) != 3:
                return ''
        if c != 'e' or code[0] != 'v':
            return ''
        return code

    def after_parse_diagnosis(self):
        t = OrderedDict.fromkeys(self.admission_codes.keys())
        for admission_id, codes in self.admission_codes.items():
            t[admission_id] = list(set(codes))
        self.admission_codes = t
