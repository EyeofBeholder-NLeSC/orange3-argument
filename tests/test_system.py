"""System testing of the whole package."""
from orangearg.argument.miner import reader, chunker, processor, miner

from .conftest import TEST_DATA_FOLDER


def test_system():
    """Systen testing of the whole package."""
    fpath = TEST_DATA_FOLDER / "reviews.json"

    # reader
    df_arguments = reader.read_json_file(fpath=fpath)
    df_arguments = df_arguments.dropna().reset_index(drop=True)
    df_arguments = df_arguments.rename(
        columns={"reviewText": "argument", "overall": "score"}
    )
    assert df_arguments["argument"].dtype == "object"
    assert df_arguments["score"].dtype == "int64"

    # chunker
    arguments = df_arguments["argument"]
    arg_scores = df_arguments["score"]
    chunk_arg_ids, chunks = chunker.get_chunk(docs=arguments)
    chunk_p_scores = chunker.get_chunk_polarity_score(chunks=chunks)
    chunk_topics, chunk_embeds, df_topics = chunker.get_chunk_topic(chunks=chunks)
    chunk_ranks = chunker.get_chunk_rank(arg_ids=chunk_arg_ids, embeds=chunk_embeds)
    assert all(-1 <= s <= 1 for s in chunk_p_scores)
    assert all(0 <= r <= 1 for r in chunk_ranks)
    assert (
        len(chunks)
        == len(chunk_arg_ids)
        == len(chunk_p_scores)
        == len(chunk_topics)
        == len(chunk_ranks)
        == chunk_embeds.shape[0]
    )

    df_chunks = chunker.get_chunk_table(
        arg_ids=chunk_arg_ids,
        chunks=chunks,
        p_scores=chunk_p_scores,
        topics=chunk_topics,
        ranks=chunk_ranks,
    )

    # processor
    arg_topics = processor.get_argument_topics(
        arg_ids=chunk_arg_ids, topics=chunk_topics
    )
    arg_sentiments = processor.get_argument_sentiment(
        arg_ids=chunk_arg_ids, ranks=chunk_ranks, p_scores=chunk_p_scores
    )
    arg_coherences = processor.get_argument_coherence(
        scores=arg_scores, sentiments=arg_sentiments
    )
    assert all(0 <= s <= 1 for s in arg_sentiments)
    assert all(0 < c <= 1 for c in arg_coherences)
    assert (
        len(arg_topics)
        == len(arg_sentiments)
        == len(arg_coherences)
        == df_arguments.shape[0]
    )

    df_arguments_processed = processor.update_argument_table(
        df_arguments=df_arguments,
        topics=arg_topics,
        sentiments=arg_sentiments,
        coherences=arg_coherences,
    )

    # miner
    last_topic = df_topics.iloc[-1]["topic"]
    arg_selection = miner.select_by_topic(data=df_arguments_processed, topic=last_topic)
    edges = miner.get_edges(data=arg_selection)
    weights = miner.get_edge_weights(data=arg_selection, edges=edges)
    assert all(i in arg_selection.index for edge in edges for i in edge)
    assert all(-1 <= w <= 1 for w in weights)

    df_edges = miner.get_edge_table(edges=edges, weights=weights)

    labels = miner.get_node_labels(
        indices=arg_selection.index.tolist(),
        sources=df_edges["source"].tolist(),
        targets=df_edges["target"].tolist(),
    )
    assert all(l in ["supportive", "defeated"] for l in labels)
    assert len(labels) == arg_selection.shape[0]

    df_nodes = miner.get_node_table(
        arg_ids=arg_selection["argument_id"].tolist(),
        arguments=arg_selection["argument"].tolist(),
        scores=arg_selection["score"].tolist(),
        labels=labels,
    )
