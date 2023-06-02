from pathlib import Path
from dataclasses import dataclass
from typing import List
import json

from fire import Fire
from tqdm import tqdm


@dataclass
class Annotations:
    split: str
    questions: List[str]
    answers: List[str]


def main(in_dir: str, out_dir: str):
    splits = ['train', 'dev-0', 'test-A']
    split_to_due = {'train': 'train', 'dev-0': 'dev', 'test-A': 'test'}
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir()

    out_split_dirs = {split: out_dir / split for split in split_to_due.values()}
    for d in out_split_dirs.values():
        d.mkdir()

    annotations_dict = {}
    for split in splits:
        expected_path = in_dir /split / 'expected.tsv'
        if expected_path.exists():
            with open(in_dir / split / 'in.tsv') as in_f, open(in_dir /split / 'expected.tsv') as expected_f:
                for in_line, expected_line in zip(in_f, expected_f):
                    file, question, _ = in_line.split('\t')
                    question = question.strip()
                    answer = expected_line.strip()
                    if file in annotations_dict:
                        assert annotations_dict[file].split == split_to_due[split]
                        annotations_dict[file].questions.append(question)
                        annotations_dict[file].answers.append(answer)
                    else:
                        annotations_dict[file] = Annotations(split_to_due[split], [question], [answer])
        else:
            with open(in_dir / split / 'in.tsv') as in_f:
                for in_line in in_f:
                    file, question, _ = in_line.split('\t')
                    question = question.strip()
                    if file in annotations_dict:
                        assert annotations_dict[file].split == split_to_due[split]
                        annotations_dict[file].questions.append(question)
                    else:
                        annotations_dict[file] = Annotations(split_to_due[split], [question], [])

    split_document = {split: open(d / 'document.jsonl', 'w') for split, d in out_split_dirs.items()}
    split_documents_contents = {split: open(d / 'documents_content.jsonl', 'w') for split, d in out_split_dirs.items()}

    question_id = 0
    with open(out_dir / 'document.jsonl', 'w') as document_out,\
            open(out_dir / 'documents_content.jsonl', 'w') as documents_content_out:
        for file, annotations in tqdm(annotations_dict.items()):
            name = '.'.join(file.split('.')[0:-1])
            annotations_due = []
            if annotations.answers:
                for question, answer in zip(annotations.questions, annotations.answers):
                    annotations_due.append({'id': str(question_id), 'key': question, 'values': [{'value': answer, 'value_variants': [answer]}]})
                    question_id += 1
            else:
                assert annotations.split == 'test'
                answer_mock = 'answer_mock'
                for question in annotations.questions:
                    annotations_due.append({'id': str(question_id), 'key': question, 'values': [{'value': answer_mock, 'value_variants': [answer_mock]}]})
                    question_id += 1
            file_dict = {'name': name, 'language': 'en', 'annotations': annotations_due, 'split': annotations.split}
            document_out.write(f'{json.dumps(file_dict)}\n')
            split_document[annotations.split].write(f'{json.dumps(file_dict)}\n')

            with open(in_dir / 'documents-ocr' / f'{name}.jpg.json') as ocr_layer_f:
                ocr_layer = json.load(ocr_layer_f)
            text = ' '.join(ocr_layer['tokens'])
            contents_dict = {'name': name, 'contents': [{'tool_name': 'microsoft_cv', 'text': text, 'common_format': ocr_layer}]}
            documents_content_out.write(f'{json.dumps(contents_dict)}\n')
            split_documents_contents[annotations.split].write(f'{json.dumps(contents_dict)}\n')

    for f in split_document.values():
        f.close()

    for f in split_documents_contents.values():
        f.close()


if __name__ == '__main__':
    Fire(main)
