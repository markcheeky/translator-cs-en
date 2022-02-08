from datetime import datetime

from pathlib import Path
from typing import Optional
from collections import OrderedDict

import typer


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

    from joblib import Parallel, delayed
    import pandas as pd
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
) -> None:

    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    device = gpu if gpu is not None else -1

    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    translate = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

    while True:
        user_input = input("Type your czech sentence (leave empty to exit): ")
        if user_input == "":
            break
        print(translate(user_input)[0]["translation_text"])


@app.command()
def train(
    output_dir: Path,
    train_src_path: Path,
    train_tgt_path: Path,
    valid_src_path: Path,
    valid_tgt_path: Path,
    steps: int = typer.Option(...),
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
        max_steps=steps,
        lr=lr,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        save_steps=save_steps,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        device=device,
    )


def main():
    app()
