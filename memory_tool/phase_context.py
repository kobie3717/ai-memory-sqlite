"""Phase detection and counterfactual scoring for memory export."""

import os
from typing import Dict, Any

PHASE_CATEGORY_WEIGHTS = {
    'debugging': {
        'error': 1.0, 'architecture': 0.8, 'decision': 0.7,
        'learning': 0.9, 'workflow': 0.5, 'preference': 0.2,
        'project': 0.3, 'pending': 0.4, 'contact': 0.1,
    },
    'coding': {
        'architecture': 0.9, 'decision': 0.8, 'workflow': 0.8,
        'error': 0.6, 'learning': 0.7, 'preference': 0.5,
        'project': 0.4, 'pending': 0.6, 'contact': 0.1,
    },
    'review': {
        'decision': 1.0, 'architecture': 0.9, 'error': 0.8,
        'learning': 0.7, 'workflow': 0.6, 'preference': 0.3,
        'project': 0.5, 'pending': 0.4, 'contact': 0.1,
    },
    'planning': {
        'decision': 1.0, 'architecture': 1.0, 'project': 0.9,
        'pending': 0.8, 'workflow': 0.6, 'preference': 0.5,
        'error': 0.4, 'learning': 0.5, 'contact': 0.3,
    },
    'general': {
        'decision': 0.8, 'architecture': 0.8, 'error': 0.7,
        'learning': 0.7, 'workflow': 0.6, 'pending': 0.6,
        'project': 0.5, 'preference': 0.4, 'contact': 0.2,
    },
}

def detect_phase() -> str:
    """Detect current session phase from env signals."""
    # UserPromptSubmit hook sets this
    task_type = os.environ.get('CLAUDE_TASK_TYPE', '').lower()
    mapping = {
        'debugging': 'debugging', 'testing': 'debugging',
        'coding': 'coding', 'devops': 'coding',
        'review': 'review',
        'planning': 'planning', 'research': 'planning',
        'monitoring': 'general',
    }
    if task_type in mapping:
        return mapping[task_type]
    return 'general'

def counterfactual_score(memory: Dict[str, Any], phase: str = 'general') -> float:
    """Score: how bad is it if Claude forgets this? 0.0-1.0"""
    weights = PHASE_CATEGORY_WEIGHTS.get(phase, PHASE_CATEGORY_WEIGHTS['general'])
    cat = memory.get('category') or 'learning'
    score = weights.get(cat, 0.3)

    # Priority boost (priority is 0-10)
    priority = memory.get('priority') or 5
    score += (priority / 10.0) * 0.2

    # Tag boosts for high-stakes content
    tags = (memory.get('tags') or '').lower()
    content = (memory.get('content') or '').lower()
    combined = tags + ' ' + content
    if any(t in combined for t in ['security', 'auth', 'payment', 'critical', 'production']):
        score += 0.25
    if any(t in combined for t in ['footgun', 'dangerous', 'never', 'always', 'must']):
        score += 0.15
    if any(t in combined for t in ['error-prevention', 'lesson', 'mistake']):
        score += 0.10

    # Invert access_count bias: rare but high-priority = high counterfactual
    access_count = memory.get('access_count') or 0
    if access_count < 3 and priority >= 7:
        score += 0.15  # Rare + important = high risk of forgetting

    # Penalize potentially-wrong memories
    if 'contradiction-flagged' in tags:
        score -= 0.30

    return min(1.0, max(0.0, score))

def survivor_score(memory: Dict[str, Any]) -> float:
    """Max counterfactual score across all phases — for survivor layer."""
    return max(counterfactual_score(memory, phase) for phase in PHASE_CATEGORY_WEIGHTS)
