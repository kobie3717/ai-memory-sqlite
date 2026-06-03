"""
Core memory operations - imports all functionality from the modularized components.
This module serves as the central hub for all memory operations.
"""

# Re-export everything from our specialized modules
from .config import *
from .database import get_db, init_db, has_vec_support
from .fsrs import *
from .importance import *
from .embedding import (
    get_embedding_model, embed_text, embed_texts_batch,
    embed_and_store, semantic_search, reindex_embeddings
)
from .utils import auto_tag, normalize, word_set, word_overlap, similarity, find_similar

# Import from all the new modular operations files
from .memory_ops import *
from .relations import *
from .graph import *
from .snapshots import *
from .sync import *
from .runs import *
from .dream import *
from .corrections import *
from .export import *
from .display import *
from .feedback import *
from .narrative import build_narrative, get_entity_stories, get_causal_chains
from .meta_learning import (
    get_current_weights, save_weights, calculate_effectiveness,
    apply_learned_weights, get_meta_stats, get_weight_history
)
from .identity import discover_traits, get_identity, save_identity_snapshot, get_identity_evolution, compare_identity_snapshots
from .focus import focus_topic, cmd_focus
from .user_model import generate_user_model, update_user_model_with_transcript, scan_user_memories, cmd_user_model
