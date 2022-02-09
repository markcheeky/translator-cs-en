import math
from collections import OrderedDict
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import List, Optional

import typer
from tqdm import tqdm

from translator.utils import chunkify

app = typer.Typer()


SRC_LANG = "cs"
TGT_LANG = "en"


@app.command()
def generate_dataset(
    file: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        default=None,
    ),
    output_src: str = typer.Option(...),
    output_tgt: str = typer.Option(...),
    read_first_n: Optional[int] = None,
    min_sample_len: int = 150,
    min_adq_score: float = 0.2,
    min_lang_score: float = 0.8,
    on_bad_lines: str = "warn",
    chunk_size: int = 1_000_000,
) -> None:

    import pandas as pd
    from joblib import Parallel, delayed

    from translator.preprocess import process_dataset

    model_name = f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"
    columns = OrderedDict(
        [
            ("id", str),
            ("adq_score", float),
            ("src_lang_score", float),
            ("tgt_lang_score", float),
            ("src_text", str),
            ("tgt_text", str),
        ]
    )

    print(f"start: {datetime.now()}")
    print("starting to read and process the data")

    chunks = pd.read_csv(
        file,
        sep="\t",
        names=list(columns.keys()),
        nrows=read_first_n,
        on_bad_lines=on_bad_lines,
        dtype=columns,
        chunksize=chunk_size,
    )

    list(
        Parallel(n_jobs=-1)(
            delayed(process_dataset)(
                df,
                model_name,
                min_sample_len,
                min_adq_score,
                min_lang_score,
                output_src,
                output_tgt,
                i,
            )
            for i, df in enumerate(list(chunks))
        )
    )

    print("processing done")

    print(f"end: {datetime.now()}")


@app.command()
def translate(
    model_path: Path,
    gpu: Optional[int] = None,
    files: List[Path] = [],
    batch_size: int = 1,
    read_first_n: Optional[int] = None,
    translated_filename_suffix: str = "-translated",
) -> None:

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    if len(files) == 0:
        output_files = [Path(str(file) + translated_filename_suffix) for file in files]
        for file, output_file in zip(files, output_files):
            if not file.exists() or not file.is_file():
                print(f"file '{file}' does not exist or is a directory. Aborting.")
                return
            if output_file.exists():
                print(f"output file '{output_file}' already exists. Aborting.")
                return

    device = gpu if gpu is not None else -1

    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    translate = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

    if len(files) == 0:
        # interactive mode
        while True:
            user_input = input("Type a czech sentence (leave empty to exit): ")
            if user_input == "":
                return
            print(translate(user_input)[0]["translation_text"])
    else:
        # translating documents
        for file, output_file in zip(files, output_files):
            print()
            print(f"translating '{file}' to '{output_file}'")

            if read_first_n is not None:
                line_count = read_first_n
            else:
                with open(file, "r") as f:
                    line_count = sum(1 for line in f)

            batch_count = math.ceil(line_count / batch_size)

            with open(file, "r") as f, open(output_file, "a") as out:
                batches = chunkify((line.rstrip() for line in islice(f, 0, line_count)), batch_size)
                output_batches = map(translate, tqdm(batches, total=batch_count))
                for output_batch in output_batches:
                    for output in output_batch:
                        out.write(output["translation_text"].replace("\n", "") + "\n")


@app.command()
def train(
    output_dir: Path,
    train_src_path: Path,
    train_tgt_path: Path,
    valid_src_path: Path,
    valid_tgt_path: Path,
    epochs: int = typer.Option(...),
    batch_size: int = typer.Option(...),
    grad_acc_steps: int = typer.Option(...),
    save_steps: int = typer.Option(...),
    starting_point: Optional[Path] = typer.Option(None),
    lr: float = 0.000025,
    gpu: Optional[int] = None,
) -> None:

    from translator.train import train_model

    device = gpu if gpu is not None else "cpu"

    train_model(
        starting_point=starting_point,
        output_dir=output_dir,
        train_src_path=train_src_path,
        train_tgt_path=train_tgt_path,
        valid_src_path=valid_src_path,
        valid_tgt_path=valid_tgt_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        save_steps=save_steps,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        device=device,
    )


@app.command()
def eval(
    actual: Path = typer.Option(...),
    reference: Path = typer.Option(...),
    read_first_n: Optional[int] = None,
) -> None:

    from adaptor.evaluators.generative import BLEU, METEOR, ROUGE

    with open(actual, "r") as file:
        actual_strings = [line.rstrip() for line in islice(file, 0, read_first_n)]

    with open(reference, "r") as file:
        reference_strings = [line.rstrip() for line in islice(file, 0, read_first_n)]

    if len(actual_strings) != len(reference_strings):
        print("files have difference number of lines.")
        print("Check if you put the right paths as arguments.")
        max_lines = max(len(actual_strings), len(reference_strings))
        actual_strings = actual_strings[:max_lines]
        reference_strings = reference_strings[:max_lines]
        print("files were truncated to the same line count")

    bleu = BLEU(additional_sep_char="_").evaluate_str(reference_strings, actual_strings)
    rouge_l = ROUGE(additional_sep_char="_").evaluate_str(reference_strings, actual_strings)
    meteor = METEOR(additional_sep_char="_").evaluate_str(reference_strings, actual_strings)
    print("Bleu:", bleu)
    print("Rouge-L:", rouge_l)
    print("Meteor:", meteor)


def main():
    app()
