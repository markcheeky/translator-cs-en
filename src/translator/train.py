from pathlib import Path
from typing import Optional, Union

import torch
from adaptor.lang_module import LangModule

from adaptor.adapter import Adapter, AdaptationArguments
from adaptor.lang_module import LangModule
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.utils import AdaptationArguments, StoppingStrategy
from adaptor.schedules import SequentialSchedule
from adaptor.evaluators.generative import BLEU, ROUGE


def train_model(
    starting_point: Optional[Path],
    train_src_path: Path,
    train_tgt_path: Path,
    valid_src_path: Path,
    valid_tgt_path: Path,
    output_dir: Path,
    device: Union[int, str],
    src_lang: str,
    tgt_lang: str,
    lr: float,
    batch_size: int,
    grad_acc_steps: int,
    save_steps: int,
    epochs: int,
):

    if starting_point is None:
        starting_point = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    lang_model = LangModule(starting_point).to(device)

    metrics = [
        BLEU(additional_sep_char="_"),
        ROUGE(additional_sep_char="_"),
    ]

    objective = Sequence2Sequence(
        lang_model,
        texts_or_path=str(train_src_path),
        labels_or_path=str(train_tgt_path),
        val_texts_or_path=str(valid_src_path),
        val_labels_or_path=str(valid_tgt_path),
        batch_size=batch_size,
        val_evaluators=metrics,
    )

    training_args = AdaptationArguments(
        output_dir=output_dir,
        learning_rate=lr,
        stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS,
        do_train=True,
        do_eval=True,
        num_train_epochs=epochs,
        gradient_accumulation_steps=grad_acc_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
    )

    schedule = SequentialSchedule([objective], args=training_args)

    adapter = Adapter(lang_model, schedule, args=training_args)
    adapter.train()
    adapter.save_model(output_dir)
