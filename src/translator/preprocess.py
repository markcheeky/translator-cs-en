from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple


import pandas as pd

from transformers import PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class Chunk:
    src_text: str
    tgt_text: str
    src_token_count: int
    tgt_token_count: int

    def __add__(self, other: "Chunk") -> "Chunk":
        return Chunk(
            self.src_text + other.src_text,
            self.tgt_text + other.tgt_text,
            self.src_token_count + other.src_token_count,
            self.tgt_token_count + other.tgt_token_count,
        )


def group_to_fit(
    src_texts: Iterable[str],
    tgt_texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: Optional[int] = None,
    endings: Tuple[str, ...] = (".", "?", "!", ";", "…"),
) -> Generator[Tuple[str, str], None, None]:
    """Groups sentences of one document into several parts so that each
    part fits into model input size after tokenization (in both languages)

    Sentences that have more tokens than max_tokens on their own are skipped
    """

    if max_tokens is None:
        max_tokens = tokenizer.model_max_length

    part = Chunk("", "", 0, 0)
    sep_space = Chunk(" ", " ", 1, 1)
    sep_dot = Chunk(". ", ". ", 2, 2)
    for src_text, tgt_text in zip(src_texts, tgt_texts):

        src_curr_count = len(tokenizer.tokenize(src_text))
        with tokenizer.as_target_tokenizer():
            tgt_curr_count = len(tokenizer.tokenize(tgt_text))

        curr = Chunk(src_text, tgt_text, src_curr_count, tgt_curr_count)

        if part.src_text == "":
            new = part + curr
        elif part.src_text.endswith(endings):
            new = part + sep_space + curr
        else:
            new = part + sep_dot + curr

        if new.src_token_count < max_tokens and new.tgt_token_count < max_tokens:
            part = new
        else:
            yield part.src_text, part.tgt_text
            part = Chunk(src_text, tgt_text, 0, 0)

    if part.src_text != "":
        yield part.src_text, part.tgt_text


def process_dataset(
    df: pd.DataFrame,
    model_name: str,
    min_sample_len: float,
    min_adq_score: float,
    min_lang_score: float,
    output_src: str,
    output_tgt: str,
    chunk_num: int,
) -> pd.DataFrame:

    df[["doc_id", "text_num"]] = df.id.str.extract(r"^(.*)-s(\d+)$")

    used_cols = {
        "doc_id",
        "text_num",
        "src_text",
        "tgt_text",
        "adq_score",
        "src_lang_score",
        "tgt_lang_score",
    }
    assert set(df.columns).issuperset(used_cols)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df.src_text = df.src_text.str.replace("\n", " ").str.strip()
    df.tgt_text = df.tgt_text.str.replace("\n", " ").str.strip()

    prepared = (
        df.dropna()
        .drop_duplicates("src_text")
        .drop_duplicates("tgt_text")
        .groupby("doc_id")
        .filter(
            lambda group: sum(map(len, group.src_text)) > min_sample_len
            and group.adq_score.mean() > min_adq_score
            and group.src_lang_score.mean() > min_lang_score
            and group.tgt_lang_score.mean() > min_lang_score
        )
        .groupby("doc_id")
        .apply(lambda group: list(group_to_fit(group.src_text, group.tgt_text, tokenizer)))
        .explode()
        .apply(pd.Series)
        .rename(columns={0: "src_text", 1: "tgt_text"})
    )

    output_src = f"{output_src}-{chunk_num:04}"
    output_tgt = f"{output_tgt}-{chunk_num:04}"

    with open(output_src, "w") as file_src, open(output_tgt, "w") as file_tgt:
        for src_text, tgt_text in prepared.itertuples(index=False):
            file_src.write(src_text + "\n")
            file_tgt.write(tgt_text + "\n")

    print(f"chunk {chunk_num:04} done")
