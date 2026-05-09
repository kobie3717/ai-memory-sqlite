"""
Unsupervised SimCSE fine-tuning for AI-IQ embedding model.

SimCSE (Simple Contrastive Sentence Embeddings) trains a better encoder
by treating each sentence as its own positive pair under different dropout
masks. No labels required — uses your memory corpus as training data.

Reference: https://arxiv.org/abs/2104.08821 (Gao et al., 2021)

Usage:
    from memory_tool.ssl_finetune import finetune_on_memories
    finetune_on_memories()  # fine-tunes and replaces ONNX model
"""

import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from .config import MODEL_DIR, get_logger
from .database import get_db

logger = get_logger(__name__)

BASE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32


def _load_memory_corpus(min_length: int = 20) -> List[str]:
    """Load all active memory content as training corpus."""
    conn = get_db()
    rows = conn.execute(
        "SELECT content FROM memories WHERE active = 1 AND length(content) >= ?",
        (min_length,)
    ).fetchall()
    conn.close()
    texts = [r['content'] for r in rows if r['content'] and r['content'] != '[erased]']
    logger.info(f"Loaded {len(texts)} memories for SimCSE training")
    return texts


def _train_simcse(
    texts: List[str],
    output_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Unsupervised SimCSE: treat each sentence as its own positive pair.
    Uses MultipleNegativesRankingLoss which is the NT-Xent loss used in SimCSE.
    Dropout in the encoder creates two different representations of the same sentence.
    """
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.readers import InputExample
    from torch.utils.data import DataLoader

    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    model = SentenceTransformer(BASE_MODEL_ID)

    # SimCSE: each example is (sentence, sentence) — same text, different dropout
    train_examples = [InputExample(texts=[t, t]) for t in texts]
    logger.info(f"Training examples: {len(train_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=min(batch_size, len(train_examples)),
    )

    # MultipleNegativesRankingLoss = InfoNCE/NT-Xent — the SimCSE objective
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, len(train_dataloader) // 5)
    logger.info(f"Training {epochs} epoch(s), warmup={warmup_steps} steps")

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # Save as HuggingFace model (needed for ONNX export)
    model.save(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")


def _export_to_onnx(hf_model_dir: str, onnx_output_path: str) -> Optional[str]:
    """
    Export fine-tuned model to ONNX via HuggingFace Optimum.
    Optimum handles dynamic batch/sequence axes correctly for BERT-family models.
    Returns path to .data sidecar if one was produced, else None.
    """
    from optimum.exporters.onnx import main_export

    export_dir = onnx_output_path + "_export"
    os.makedirs(export_dir, exist_ok=True)

    logger.info("Exporting to ONNX via optimum (dynamic axes)...")
    main_export(
        model_name_or_path=hf_model_dir,
        output=export_dir,
        task="feature-extraction",
        monolith=True,
        no_post_process=True,
    )

    # optimum writes export_dir/model.onnx (and optionally model.onnx.data)
    exported_onnx = os.path.join(export_dir, "model.onnx")
    if not os.path.exists(exported_onnx):
        raise FileNotFoundError(f"Optimum did not produce {exported_onnx}")

    onnx_dir = os.path.dirname(onnx_output_path)
    os.makedirs(onnx_dir, exist_ok=True)
    shutil.copy2(exported_onnx, onnx_output_path)
    logger.info(f"ONNX model exported to {onnx_output_path}")

    exported_data = os.path.join(export_dir, "model.onnx.data")
    data_dest = onnx_output_path + ".data"
    if os.path.exists(exported_data):
        shutil.copy2(exported_data, data_dest)
        logger.info(f"External data file: {data_dest}")
        shutil.rmtree(export_dir, ignore_errors=True)
        return data_dest

    shutil.rmtree(export_dir, ignore_errors=True)
    return None


def finetune_on_memories(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_corpus_size: int = 10,
    dry_run: bool = False,
) -> bool:
    """
    Full SimCSE pipeline:
    1. Load memory corpus from DB
    2. Fine-tune all-MiniLM-L6-v2 with SimCSE objective
    3. Export fine-tuned model to ONNX
    4. Backup existing ONNX, replace with fine-tuned version
    5. Log outcome

    Returns True on success.
    """
    texts = _load_memory_corpus()

    if len(texts) < min_corpus_size:
        logger.warning(
            f"Only {len(texts)} memories found (min={min_corpus_size}). "
            "Add more memories before fine-tuning."
        )
        return False

    if dry_run:
        logger.info(f"Dry run: would train on {len(texts)} memories, {epochs} epochs")
        return True

    with tempfile.TemporaryDirectory(prefix="aiiq_ssl_") as tmpdir:
        hf_out = os.path.join(tmpdir, "finetuned")
        onnx_out = os.path.join(tmpdir, "model.onnx")

        try:
            _train_simcse(texts, hf_out, epochs=epochs, batch_size=batch_size)
        except Exception as e:
            logger.error(f"SimCSE training failed: {e}")
            raise

        try:
            data_file = _export_to_onnx(hf_out, onnx_out)
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

        # Backup existing model (both .onnx and .data sidecar if present)
        existing_onnx = MODEL_DIR / "onnx" / "model.onnx"
        backup_path = MODEL_DIR / "onnx" / "model.onnx.pre-ssl"
        if existing_onnx.exists():
            shutil.copy2(str(existing_onnx), str(backup_path))
            logger.info(f"Backed up existing model to {backup_path}")
        existing_data = MODEL_DIR / "onnx" / "model.onnx.data"
        if existing_data.exists():
            shutil.copy2(str(existing_data), str(MODEL_DIR / "onnx" / "model.onnx.data.pre-ssl"))

        # Replace with fine-tuned
        (MODEL_DIR / "onnx").mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_out, str(existing_onnx))
        logger.info(f"Replaced ONNX model at {existing_onnx}")

        # Copy external data sidecar (PyTorch 2.10 dynamo exporter always creates one for large models)
        if data_file and os.path.exists(data_file):
            shutil.copy2(data_file, str(existing_data))
            logger.info(f"Copied ONNX data sidecar to {existing_data}")
        elif existing_data.exists():
            # Fine-tuned model has no sidecar but old one exists — remove to avoid stale reference
            existing_data.unlink()
            logger.info("Removed stale ONNX data sidecar (fine-tuned model uses single file)")

    return True
