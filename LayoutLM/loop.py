import pytorch_lightning as pl
from transformers import AutoProcessor, AutoModelForQuestionAnswering, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from math import ceil, inf
import regex
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


LAYOUTLM_MODEL = "microsoft/layoutlmv3-large"
DATASET_PATH = "/PATH/TO/CLONED/CHALLENGE"
LOG_DIR = '/WHERE/TO/STORE/RESULTS'
CHECKPOINT_PATH = LOG_DIR + '/model'
LR = 3e-5
VAL_CHECK_INTERVAL = 0.2
MAX_EPOCHS = 10
PATIENCE = 5
BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 2
SEED = 4
LOADER_WORKERS = 4


class LayoutLMModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(LAYOUTLM_MODEL)
        self.model = AutoModelForQuestionAnswering.from_pretrained(LAYOUTLM_MODEL)

    @classmethod
    def from_best_checkpoint(cls):
        checkpoint_list = list(Path(CHECKPOINT_PATH).iterdir())
        assert len(checkpoint_list) == 1
        return cls.load_from_checkpoint(checkpoint_list[0])

    def forward(self, encoding, start_positions, end_positions):
        pass

    def training_step(self, batch, batch_idx):
        ans_begins, ans_ends = batch['answer_spans']
        del batch['answer_spans']
        outputs = self.model(**batch, start_positions=ans_begins, end_positions=ans_ends)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        ans_begins, ans_ends = batch['answer_spans']
        del batch['answer_spans']
        outputs = self.model(**batch, start_positions=ans_begins, end_positions=ans_ends)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        # Predictions only for batch size = 1!
        fname = batch.pop('fname')
        question = batch.pop('question')
        assert len(fname) == 1
        assert len(question) == 1
        outputs = self.model(**batch)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = batch['input_ids'][0, answer_start_index: answer_end_index + 1]
        answer = self.tokenizer.decode(predict_answer_tokens).strip()
        return fname[0].strip(), question[0].strip(), answer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return optimizer


class ArxivDatasetBase(Dataset):
    def __init__(self, image_paths, words, bboxes, questions):
        assert len(image_paths) == len(words)
        assert len(image_paths) == len(questions)
        assert len(image_paths) == len(bboxes)
        self.image_paths = image_paths
        self.words = words
        self.questions = questions
        self.bboxes = bboxes
        self.processor = AutoProcessor.from_pretrained(LAYOUTLM_MODEL, apply_ocr=False)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _truncate_question(question, words_before=17, words_after=7):
        question_words = question.split(' ')
        mask_position = [i for i, x in enumerate(question_words) if '<mask>' in x]
        mask_position = mask_position[0]
        begin = max(0, mask_position - words_before)
        end = min(len(question_words), mask_position + words_after + 1)
        question_truncated = ' '.join(question_words[begin: end])
        if len(question_truncated) > 500:
            print('ech')
            question_truncated = question_truncated[:500]
        return question_truncated

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        question = self._truncate_question(self.questions[idx])
        encoding = self.processor(img, question, self.words[idx], boxes=self.bboxes[idx], return_tensors="pt",
                                  padding='max_length', truncation="only_second")
        encoding.data = {k: torch.squeeze(v, dim=0) for k, v in encoding.data.items()}
        return encoding


class ArxivDatasetTrain(ArxivDatasetBase):
    def __init__(self, image_paths, words, bboxes, questions, answer_spans):
        super(ArxivDatasetTrain, self).__init__(image_paths, words, bboxes, questions)
        assert len(image_paths) == len(answer_spans)
        self.answer_spans = answer_spans

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        encoding = super(ArxivDatasetTrain, self).__getitem__(idx)
        start_word_tokens = encoding.word_to_tokens(self.answer_spans[idx][0], sequence_index=1)
        end_word_tokens = encoding.word_to_tokens(self.answer_spans[idx][1], sequence_index=1)
        if start_word_tokens and end_word_tokens:
            ans_start_token = start_word_tokens.start
            ans_end_token = end_word_tokens.end - 1
        else:
            ans_start_token, ans_end_token = 0, 0
        encoding.data['answer_spans'] = (ans_start_token, ans_end_token)
        return encoding.data


class ArxivDatasetTest(ArxivDatasetBase):
    def __getitem__(self, idx):
        encoding = super(ArxivDatasetTest, self).__getitem__(idx)
        question_padded = self.questions[idx].ljust(10000)[:10000]
        fname = self.image_paths[idx].name
        fname_padded = fname.ljust(100)[:100]
        encoding.data['fname'] = fname_padded
        encoding.data['question'] = question_padded
        return encoding.data


class ArxivDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = DATASET_PATH, batch_size: int = BATCH_SIZE):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._bs = batch_size

    def _load_OCR(self, name):
        with open(self._data_dir / 'documents-ocr' / f'{name}.json') as ocr_f:
            ocr = json.load(ocr_f)
        return ocr

    def _normalize_bbox(self, bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def _read_input(self, line_in):
        fname, question, question_type = line_in.split('\t')
        if question_type.strip() == 'extractive':
            img_path = self._data_dir / 'images' / fname
            ocr = self._load_OCR(fname)
            tokens = ocr['tokens']
            x_max, y_max = ocr['structures']['pages']['positions'][0][2:4]
            bboxes = [self._normalize_bbox(bbox, x_max, y_max) for bbox in ocr['positions']]
            question = question.strip()
            return img_path, tokens, bboxes, question
        else:
            return None

    def words_to_text(self, words):
        content = ''
        chars_to_words = []
        for i, w in enumerate(words):
            content = content + w + ' '
            chars_to_words.extend([i] * (len(w) + 1))
        return content, chars_to_words

    def _autotag_char(self, answer, words):
        MISMATCH_FRACTION = 0.2
        document, chars_to_words = self.words_to_text(words)
        mismatch_allowed = ceil(len(answer) * MISMATCH_FRACTION)
        best_match = None
        best_score = inf
        best_position = inf
        error_allowance = '{e<=' + str(mismatch_allowed )+ '}'
        ans_regex = r'(?<=(^|\W))' + f'({regex.escape(answer)}){error_allowance}' + r'(?=($|\W))'
        for match in regex.finditer(ans_regex, document, regex.IGNORECASE, overlapped=True):
            score = sum(match.fuzzy_counts)
            if score < best_score:
                best_match = match
                best_score = score
                best_position = match.span()[0]
            elif score == best_score:
                if match.span()[0] < best_position:
                    best_match = match
                    best_position = match.span()[0]
            # print('a')
        if not best_match:
            return None
        else:
            start_char, end_char = best_match.span()
            return chars_to_words[start_char], chars_to_words[end_char - 1]

    def _read_split_with_ans(self, split):
        image_paths = []
        words = []
        bboxes_list = []
        questions = []
        ans_positions = []
        with open(self._data_dir / split / 'in.tsv') as in_f, \
                open(self._data_dir / split / 'expected.tsv') as expected_f:
            for i, (line_in, expected_ans) in tqdm(enumerate(zip(in_f, expected_f))):
                model_in = self._read_input(line_in)
                if model_in:
                    expected_ans = expected_ans.strip()
                    img_path, tokens, bboxes, question = model_in
                    ans_position = self._autotag_char(expected_ans, tokens)
                    if ans_position:
                        image_paths.append(img_path)
                        words.append(tokens)
                        bboxes_list.append(bboxes)
                        questions.append(question)
                        ans_positions.append(ans_position)
        return image_paths, words, bboxes_list, questions, ans_positions

    def _read_test_split(self):
        image_paths = []
        words = []
        bboxes_list = []
        questions = []
        with open(self._data_dir / 'test-A' / 'in.tsv') as in_f:
            for i, line_in in tqdm(enumerate(in_f)):
                model_in = self._read_input(line_in)
                if model_in:
                    img_path, tokens, bboxes, question = model_in
                    image_paths.append(img_path)
                    words.append(tokens)
                    bboxes_list.append(bboxes)
                    questions.append(question)
        return image_paths, words, bboxes_list, questions

    def setup(self, stage: str) -> None:
        self.train_data = self._read_split_with_ans('train')
        self.dev_data = self._read_split_with_ans('dev-0')
        self.test_data = self._read_test_split()

    def train_dataloader(self):
        dataset = ArxivDatasetTrain(*self.train_data)
        return DataLoader(dataset, batch_size=self._bs, shuffle=False, num_workers=LOADER_WORKERS)

    def val_dataloader(self):
        dataset = ArxivDatasetTrain(*self.dev_data)
        return DataLoader(dataset, batch_size=self._bs, shuffle=False, num_workers=LOADER_WORKERS)

    def predict_dataloader(self):
        dataset = ArxivDatasetTest(*self.test_data)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=LOADER_WORKERS)


def save_predictions(predictions):
    with open(Path(LOG_DIR) / 'predictions.tsv', 'w') as out_file:
        for pred_tuple in predictions:
            pred_line = '\t'.join(pred_tuple)
            out_file.write(f'{pred_line}\n')


def main():
    pl.seed_everything(SEED)
    assert not Path(LOG_DIR).exists()
    logger = CSVLogger(LOG_DIR, name='layoutlm_baseline')
    model = LayoutLMModel()
    datamodule = ArxivDataModule()
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_PATH, save_top_k=1, monitor="val_loss")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=MAX_EPOCHS, logger=logger, val_check_interval=VAL_CHECK_INTERVAL,
                         strategy="ddp", accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
                         callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, datamodule=datamodule)

    best_model = LayoutLMModel.from_best_checkpoint()
    predict_trainer = pl.Trainer(accelerator='gpu', devices=1)
    predictions = predict_trainer.predict(best_model, datamodule=datamodule)
    save_predictions(predictions)


if __name__ == '__main__':
    main()
