from datetime import datetime

from pathlib import Path
from typing import Optional
from collections import OrderedDict

import pandas as pd
from joblib import Parallel, delayed
import typer

from translator.preprocess import process_dataset

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
    output_src: str = typer.Option(...),
    output_tgt: str = typer.Option(...),
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


if __name__ == "__main__":
    app()
