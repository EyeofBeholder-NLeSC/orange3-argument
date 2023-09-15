"""System testing of the whole package."""
from orangecontrib.argument.miner import reader, chunker, processor, miner

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

    # chunker
    arguments = df_arguments["argument"]
    arg_scores = df_arguments["score"]
    chunk_arg_ids, chunks = chunker.get_chunk(docs=arguments)
    chunk_p_scores = chunker.get_chunk_polarity_score(chunks=chunks)
    chunk_topics, chunk_embeds, df_topics = chunker.get_chunk_topic(chunks=chunks)

    chunk_ranks = chunker.get_chunk_rank(arg_ids=chunk_arg_ids, embeds=chunk_embeds)
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
    df_edges = miner.get_edge_table(edges=edges, weights=weights)

    labels = miner.get_node_labels(
        indices=arg_selection.index.tolist(),
        sources=df_edges["source"].tolist(),
        targets=df_edges["target"].tolist(),
    )
    df_nodes = miner.get_node_table(
        arg_ids=arg_selection["argument_id"].tolist(),
        arguments=arg_selection["argument"].tolist(),
        scores=arg_selection["score"].tolist(),
        labels=labels,
    )
