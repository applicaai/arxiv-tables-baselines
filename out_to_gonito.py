from pathlib import Path
from dataclasses import dataclass
from typing import List
import json

from fire import Fire


@dataclass
class Annotations:
    split: str
    questions: List[str]
    answers: List[str]


def add_to_answer_dict(answer_dict, fname, question, answer, question_len):
    if len(question) > question_len:
        question = question[:question_len]
    if (fname, question) in answer_dict:
        answer_dict[(fname, question)].append(answer)
        if answer_dict[(fname, question)][0] != answer:
            print('Some problems in out order because of truncated questions in memmaps.')
    else:
        answer_dict[(fname, question)] = [answer]
    return answer_dict


def read_json_out(model_out_path, question_len):
    answer_dict = {}
    with open(model_out_path) as model_out_f:
        for line in model_out_f:
            out_dict = json.loads(line)
            question = out_dict['label_name']
            fname = out_dict['doc_id']
            answer = out_dict['preds']
            answer_dict = add_to_answer_dict(answer_dict, fname, question, answer, question_len)
    return answer_dict


def read_tsv_out(model_out_path, question_len):
    answer_dict = {}
    with open(model_out_path) as model_out_f:
        for line in model_out_f:
            fname, question, answer = line.split('\t')
            fname = '.'.join(fname.split('.')[0:-1])
            answer = answer.strip()
            answer_dict = add_to_answer_dict(answer_dict, fname, question, answer, question_len)
    return answer_dict


def main(model_out_path: str, gonito_in_path: str, gonito_out_path: str):
    model_out_path = Path(model_out_path)
    gonito_in_path = Path(gonito_in_path)
    gonito_out_path = Path(gonito_out_path)

    JSON_QUESTION_LEN = 80
    CSV_QUESTION_LEN = 10000
    if model_out_path.suffix in {'.jsonl', '.txt'}:
        question_len = JSON_QUESTION_LEN
        answer_dict = read_json_out(model_out_path, question_len)
    elif model_out_path.suffix == '.tsv':
        question_len = CSV_QUESTION_LEN
        answer_dict = read_tsv_out(model_out_path, question_len)
    else:
        raise RuntimeError('Wrong model output.')

    answers = []
    with open(gonito_in_path) as gonito_in_f:
        for line in gonito_in_f:
            fname, question, _ = line.split('\t')
            fname = '.'.join(fname.split('.')[0:-1])
            question = question.strip()
            if len(question) > question_len:
                question = question[:question_len]
            if (fname, question) in answer_dict:
                ans = answer_dict[(fname, question)].pop(0)
                answers.append(ans)
            else:
                answers.append('')

    with open(gonito_out_path, 'w') as gonito_out_f:
        for a in answers:
            gonito_out_f.write(f'{a}\n')


if __name__ == '__main__':
    Fire(main)
