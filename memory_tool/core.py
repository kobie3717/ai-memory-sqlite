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
