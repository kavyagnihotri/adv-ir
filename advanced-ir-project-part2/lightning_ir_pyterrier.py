from functools import cached_property
from math import ceil
from pathlib import Path
from typing import Sequence

from lightning_ir import (
    BiEncoderModule,
    BiEncoderConfig,
    BiEncoderModel,
    CrossEncoderModule,
    CrossEncoderConfig,
    CrossEncoderModel,
    SearchConfig,
    IndexConfig,
    Indexer as LightningIndexer,
    Searcher,
    IndexBatch,
    RankBatch,
    SearchBatch,
)
import torch
from more_itertools import chunked, unzip
from pandas import DataFrame, Series, concat
from pyterrier import Transformer, Indexer
from pyterrier.model import add_ranks, IterDict
from pyterrier_alpha.utils import peekable
from pyterrier_alpha.validate import (
    any as validate_any,
    any_iter as validate_any_iter,
    columns,
)
from tqdm.auto import tqdm


class BiEncoder(Indexer, Transformer):
    _model_name_or_path: str = "webis/bert-bi-encoder"
    _config: BiEncoderConfig | None
    _model: BiEncoderModel | None
    _index_dir: Path | str | None
    _index_config: IndexConfig | None
    _search_config: SearchConfig | None
    _text_field: str
    _batch_size: int
    _verbose: bool
    _use_gpu: bool

    def __init__(
        self,
        model_name_or_path: str = "webis/bert-bi-encoder",
        config: BiEncoderConfig | None = None,
        model: BiEncoderModel | None = None,
        index_dir: Path | str | None = None,
        index_config: IndexConfig | None = None,
        search_config: SearchConfig | None = None,
        text_field: str = "text",
        batch_size: int = 1000,
        verbose: bool = False,
        use_gpu: bool = True,
    ):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._config = config
        self._model = model
        self._index_dir = index_dir
        self._index_config = index_config
        self._search_config = search_config
        self._text_field = text_field
        self._batch_size = batch_size
        self._verbose = verbose
        self._use_gpu = use_gpu

    @cached_property
    def _module(self) -> BiEncoderModule:
        return BiEncoderModule(
            model_name_or_path=self._model_name_or_path,
            config=self._config,
            model=self._model,
        )

    @cached_property
    def _searcher(self) -> Searcher:
        if self._search_config is None:
            raise ValueError("search_config must be set for searching.")
        if self._index_dir is None:
            raise ValueError("index_dir must be set for indexing.")
        return self._search_config.search_class(
            index_dir=self._index_dir,
            search_config=self._search_config,
            module=self._module,
        )

    @cached_property
    def _indexer(self) -> LightningIndexer:
        if self._index_config is None:
            raise ValueError("index_config must be set for indexing.")
        if self._index_dir is None:
            raise ValueError("index_dir must be set for indexing.")
        return self._index_config.indexer_class(
            index_dir=self._index_dir
            if isinstance(self._index_dir, Path)
            else Path(self._index_dir),
            index_config=self._index_config,
            module=self._module,
            verbose=self._verbose,
        )

    @torch.no_grad()
    def _ranker_transform_batch(self, inp: DataFrame) -> DataFrame:
        queries_iter, documents_iter = unzip(
            (
                query,
                [row[self._text_field] for _, row in group.iterrows()],
            )
            for query, group in inp.groupby("query", sort=False)
        )
        queries: Sequence[str] = list(queries_iter)  # type: ignore
        documents: Sequence[Sequence[str]] = list(documents_iter)  # type: ignore

        rank_batch = RankBatch(
            queries=queries,
            docs=documents,
        )
        model_output = self._module.forward(rank_batch)

        scores = model_output.scores
        if scores is None:
            raise RuntimeError("Lightning IR did not return scores.")
        if scores.shape != (len(inp),):
            raise RuntimeError(
                f"Expected scores shape ({(len(inp),)}), but got {scores.shape}."
            )
        return inp.reset_index().assign(
            score=Series(scores.numpy(force=True)),
        )

    def _ranker_transform(self, inp: DataFrame) -> DataFrame:
        columns(inp, includes=["qid", "query", "docno", self._text_field])

        batches = chunked(inp.index, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                total=ceil(len(inp) / self._batch_size),
                desc=f"{self._model_name_or_path} rank",
                unit="batch",
            )

        output = concat(
            [self._ranker_transform_batch(inp.loc[indices]) for indices in batches]
        )

        return add_ranks(output)

    def _ranker_transform_iter(self, inp: IterDict) -> IterDict:
        peekable_inp = peekable(inp)
        with validate_any_iter(peekable_inp) as v:
            v.columns(includes=["qid", "query", "docno", self._text_field])

        batches = chunked(peekable_inp, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                desc=f"{self._model_name_or_path} rank",
                unit="batch",
            )

        for batch in batches:
            batch_df = DataFrame(batch)
            batch_df = self._ranker_transform_batch(batch_df)
            for row in batch_df.to_dict(orient="records"):
                yield {str(k): v for k, v in row.items()}

    @torch.no_grad()
    def _searcher_transform_batch(self, inp: DataFrame) -> DataFrame:
        search_batch = SearchBatch(
            query_ids=inp["qid"].tolist(),
            queries=inp["query"].tolist(),
        )
        model_output = self._module.forward(search_batch)

        scores, document_ids = self._searcher.search(model_output)

        output: DataFrame = inp.loc[inp.index.repeat(self._searcher.search_config.k)]
        return output.reset_index().assign(
            docno=Series([doc_id for doc_ids in document_ids for doc_id in doc_ids]),
            score=Series(scores.detach().numpy(force=True)),
        )

    def _searcher_transform(self, inp: DataFrame) -> DataFrame:
        if self._searcher is None:
            raise ValueError("Searcher is not available in the module.")

        columns(inp, includes=["qid", "query"])

        batches = chunked(inp.index, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                total=ceil(len(inp) / self._batch_size),
                desc=f"{self._model_name_or_path} search",
                unit="batch",
            )

        output = concat(
            [self._searcher_transform_batch(inp.loc[indices]) for indices in batches]
        )

        return add_ranks(output)

    def _searcher_transform_iter(self, inp: IterDict) -> IterDict:
        if self._searcher is None:
            raise ValueError("Searcher is not available in the module.")

        peekable_inp = peekable(inp)
        with validate_any_iter(peekable_inp) as v:
            v.columns(includes=["qid", "query"])

        batches = chunked(peekable_inp, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                desc=f"{self._model_name_or_path} search",
                unit="batch",
            )

        for batch in batches:
            batch_df = DataFrame(batch)
            batch_df = self._searcher_transform_batch(batch_df)
            for row in batch_df.to_dict(orient="records"):
                yield {str(k): v for k, v in row.items()}

    def transform(self, inp: DataFrame) -> DataFrame:
        with validate_any(inp) as v:
            v.columns(
                includes=["qid", "query", "docno", self._text_field], mode="ranker"
            )
            v.columns(includes=["docno", self._text_field], mode="indexer")
            v.columns(includes=["query"], mode="searcher")
        if v.mode == "ranker":
            return self._ranker_transform(inp)
        elif v.mode == "searcher":
            return self._searcher_transform(inp)
        elif v.mode == "indexer":
            raise RuntimeError("Use the index() method for indexing.")
        else:
            raise ValueError(f"Unknown mode: {v.mode}")

    def transform_iter(self, inp: IterDict) -> IterDict:
        peekable_inp = peekable(inp)
        with validate_any_iter(peekable_inp) as v:
            v.columns(
                includes=["qid", "query", "docno", self._text_field], mode="ranker"
            )
            v.columns(includes=["docno", self._text_field], mode="indexer")
            v.columns(includes=["query"], mode="searcher")
        if v.mode == "ranker":
            return self._ranker_transform_iter(peekable_inp)
        elif v.mode == "searcher":
            return self._searcher_transform_iter(peekable_inp)
        elif v.mode == "indexer":
            raise RuntimeError("Use the index() method for indexing.")
        else:
            raise ValueError(f"Unknown mode: {v.mode}")

    @torch.no_grad()
    def index(self, iter: IterDict, **kwargs) -> None:
        peekable_iter = peekable(iter)
        with validate_any_iter(peekable_iter) as v:
            v.columns(includes=["docno", self._text_field])

        batches = chunked(peekable_iter, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches, desc=f"{self._model_name_or_path} index", unit="batch"
            )
        for batch in batches:
            index_batch = IndexBatch(
                doc_ids=[doc["docno"] for doc in batch],
                docs=[doc[self._text_field] for doc in batch],
            )
            model_output = self._module.forward(index_batch)
            self._indexer.add(index_batch, model_output)
        self._indexer.save()


class CrossEncoder(Indexer, Transformer):
    _model_name_or_path: str = "webis/monoelectra-base"
    _config: CrossEncoderConfig | None
    _model: CrossEncoderModel | None
    _index_dir: Path | str | None
    _text_field: str
    _batch_size: int
    _verbose: bool
    _use_gpu: bool

    def __init__(
        self,
        model_name_or_path: str = "webis/monoelectra-base",
        config: CrossEncoderConfig | None = None,
        model: CrossEncoderModel | None = None,
        index_dir: Path | str | None = None,
        text_field: str = "text",
        batch_size: int = 1000,
        verbose: bool = False,
        use_gpu: bool = True,
    ):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._config = config
        self._model = model
        self._index_dir = index_dir
        self._text_field = text_field
        self._batch_size = batch_size
        self._verbose = verbose
        self._use_gpu = use_gpu

    @cached_property
    def _module(self) -> CrossEncoderModule:
        return CrossEncoderModule(
            model_name_or_path=self._model_name_or_path,
            config=self._config,
            model=self._model,
        )

    @torch.no_grad()
    def _ranker_transform_batch(self, inp: DataFrame) -> DataFrame:
        queries_iter, documents_iter = unzip(
            (
                query,
                [row[self._text_field] for _, row in group.iterrows()],
            )
            for query, group in inp.groupby("query", sort=False)
        )
        queries: Sequence[str] = list(queries_iter)  # type: ignore
        documents: Sequence[Sequence[str]] = list(documents_iter)  # type: ignore

        rank_batch = RankBatch(
            queries=queries,
            docs=documents,
        )
        model_output = self._module.forward(rank_batch)

        scores = model_output.scores
        if scores is None:
            raise RuntimeError("Lightning IR did not return scores.")
        if scores.shape != (len(inp),):
            raise RuntimeError(
                f"Expected scores shape ({(len(inp),)}), but got {scores.shape}."
            )
        return inp.reset_index().assign(
            score=Series(scores.numpy(force=True)),
        )

    def transform(self, inp: DataFrame) -> DataFrame:
        columns(inp, includes=["qid", "query", "docno", self._text_field])

        batches = chunked(inp.index, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                total=ceil(len(inp) / self._batch_size),
                desc=f"{self._model_name_or_path} rank",
                unit="batch",
            )

        output = concat(
            [self._ranker_transform_batch(inp.loc[indices]) for indices in batches]
        )

        output = output.drop(columns=["index", "level_0"], errors="ignore").reset_index()
        return add_ranks(output)

    def transform_iter(self, inp: IterDict) -> IterDict:
        peekable_inp = peekable(inp)
        with validate_any_iter(peekable_inp) as v:
            v.columns(includes=["qid", "query", "docno", self._text_field])

        batches = chunked(peekable_inp, self._batch_size)
        if self._verbose:
            batches = tqdm(
                batches,
                desc=f"{self._model_name_or_path} rank",
                unit="batch",
            )

        for batch in batches:
            batch_df = DataFrame(batch)
            batch_df = self._ranker_transform_batch(batch_df)
            for row in batch_df.to_dict(orient="records"):
                yield {str(k): v for k, v in row.items()}
