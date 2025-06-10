from nlgmetricverse import NLGMetricverse, load_metric
import logging


def calculate_metrics(
    predictions: list[str], references: list[str], logger: logging.Logger
) -> None:
    """Function to compute NLG metrics.

    Args:
        predictions (list[str]): Containing all the predicted sentences.
        references (list[str]): Containing all the ground truth sentences.
        logger (logging.Logger): Logger.
    """

    blue1_scorer = NLGMetricverse(
        metrics=[
            load_metric(
                "bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}
            )
        ]
    )(predictions=predictions, references=references)

    bleu2_scorer = NLGMetricverse(
        metrics=[
            load_metric(
                "bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}
            )
        ]
    )(predictions=predictions, references=references)

    blue3_scorer = NLGMetricverse(
        metrics=[
            load_metric(
                "bleu", resulting_name="bleu_3", compute_kwargs={"max_order": 3}
            )
        ]
    )(predictions=predictions, references=references)

    bleu4_scorer = NLGMetricverse(
        metrics=[
            load_metric(
                "bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}
            )
        ]
    )(predictions=predictions, references=references)

    meteor_scorer = NLGMetricverse(metrics=load_metric("meteor"))(
        predictions=predictions, references=references
    )

    rouge_scorer = NLGMetricverse(metrics=[load_metric("rouge")])(
        predictions=predictions, references=references
    )

    cider_scorer = NLGMetricverse(metrics=[load_metric("cider")])(
        predictions=predictions, references=references
    )

    bertscore_scorer = NLGMetricverse(metrics=load_metric("bertscore"))(
        predictions=predictions, references=references
    )

    logger.info(f"BLEU1: {blue1_scorer['bleu_1']['score']:.3f}")
    logger.info(f"BLEU2: {bleu2_scorer['bleu_2']['score']:.3f}")
    logger.info(f"BLEU3: {blue3_scorer['bleu_3']['score']:.3f}")
    logger.info(f"BLEU4: {bleu4_scorer['bleu_4']['score']:.3f}")
    logger.info(f"METEOR: {meteor_scorer['meteor']['score']:.3f}")
    logger.info(f"ROGUE1: {rouge_scorer['rouge']['rouge1']:.3f}")
    logger.info(f"ROGUEL: {rouge_scorer['rouge']['rougeL']:.3f}")
    logger.info(f"ROGUELSUM: {rouge_scorer['rouge']['rougeLsum']:.3f}")
    logger.info(f"CIDER: {cider_scorer['cider']['score']:.3f}")
    logger.info(f"BERTSCORE: {bertscore_scorer['bertscore']['score']:.3f}")
