import torch
import os
import numpy as np
from sacrebleu.metrics import BLEU
from .trainer.train_epoch import train_epoch, validate, predict, save_model

def train(
    model,
    tokenizer,
    train_loaders,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    logger,
    config,
):

    for epoch in range(config.n_epochs):
        print(f"Start of the epoch {epoch}")
        
        train_epoch(
            model,
            tokenizer,
            optimizer,
            scheduler,
            train_loaders,
            val_loader,
            criterion,
            logger,
            epoch=epoch,
            config=config
        )

        if (epoch + 1) % config.validation == 0:
            validate(model, val_loader, logger, config)
            print("validated")

        if ((epoch + 1) % config.save_every == 0) and (config.save_strategy == 'epoch'):
            save_model(model, tokenizer, config, epoch=epoch)

        if (epoch + 1) % config.compute_metrics_every == 0:
            all_preds, all_labels = predict(
                model, tokenizer, test_loader, config, epoch=epoch
            )

            spbleu = BLEU(tokenize='flores101').corpus_score(all_preds, [all_labels]).score
            bleu = BLEU().corpus_score(all_preds, [all_labels]).score
            metrics = {
                'spbleu': spbleu,
                'bleu': bleu
            }
            print(metrics)
            for key in metrics:
                logger.add_scalar(key, float(metrics[key]))

        st = logger.get_step()
        if st >= config.max_steps:
            break

