from pathlib import Path
from typing import Optional
from collections import OrderedDict

import pandas as pd
import typer
from transformers import AutoTokenizer

from preprocess import process_dataset

app = typer.Typer()


@app.command()
def generate_dataset(
    file: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        default=None,
    ),
    output_src: Path = typer.Option(..., exists=False),
    output_tgt: Path = typer.Option(..., exists=False),
    read_first_n: Optional[int] = None,
    min_sample_len: int = 150,
    min_adq_score: float = 0.2,
    min_lang_score: float = 0.8,
    on_bad_lines: str = "warn",
    chunk_size: int = 1_000_000,
) -> None:
    src_lang = "cs"
    target_lang = "en"
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
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

    if output_src.exists() or output_tgt.exists():
        print("output file(s) already exists. aborting")
        return

    print("starting to read the data")

    df = pd.DataFrame()

    chunks = pd.read_csv(
        file,
        sep="\t",
        names=list(columns.keys()),
        nrows=read_first_n,
        on_bad_lines=on_bad_lines,
        dtype=columns,
        chunksize=chunk_size,
    )

    for chunk in chunks:
        df = pd.concat([df, chunk])
        print("#", end="")

    print()
    print("input data was read.")

    df.src_text = df.src_text.str.replace("\n", " ").str.strip()
    df.tgt_text = df.tgt_text.str.replace("\n", " ").str.strip()

    print(f"loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df[["doc_id", "text_num"]] = df.id.str.extract(r"^(.*)-s(\d+)$")

    prepared = process_dataset(df, tokenizer, min_sample_len, min_adq_score, min_lang_score)
    with open(output_src, "a") as file_src, open(output_tgt, "a") as file_tgt:
        for src_text, tgt_text in prepared.itertuples(index=False):
            file_src.write(src_text + "\n")
            file_tgt.write(tgt_text + "\n")

    print("all done.")


if __name__ == "__main__":
    app()
