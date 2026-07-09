"""Command-line interface for memory-tool."""

import sys
import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any

# Import all operations from the core module (which re-exports everything)
from . import core
from .core import *
from .config import get_logger, setup_logging

logger = get_logger(__name__)


def parse_flags(argv: List[str], start: int = 2) -> Tuple[Dict[str, Any], List[str]]:
    """Parse --flag value pairs from argv starting at position `start`.
    Returns (flags_dict, remaining_args).
    """
    flags = {}
    remaining = []
    i = start
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(argv) and not argv[i+1].startswith("--"):
                flags[key] = argv[i+1]
                i += 2
            else:
                flags[key] = True
                i += 1
        else:
            remaining.append(arg)
            i += 1
    return flags, remaining

def main() -> None:
    init_db()

    # Handle global flags for logging
    if '--verbose' in sys.argv or '-v' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        if '--verbose' in sys.argv:
            sys.argv.remove('--verbose')
        if '-v' in sys.argv:
            sys.argv.remove('-v')
    elif '--quiet' in sys.argv or '-q' in sys.argv:
        logging.getLogger().setLevel(logging.WARNING)
        if '--quiet' in sys.argv:
            sys.argv.remove('--quiet')
        if '-q' in sys.argv:
            sys.argv.remove('-q')

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    # Parse global flags for verbose/quiet BEFORE command processing
    cmd = sys.argv[1]

    # Check for --verbose or --quiet flags anywhere in argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    quiet = '--quiet' in sys.argv or '-q' in sys.argv
    setup_logging(verbose=verbose, quiet=quiet)

    if cmd == "add" and len(sys.argv) >= 4:
        category = sys.argv[2]
        content = sys.argv[3]
        flags, _ = parse_flags(sys.argv, 4)
        add_memory(
            category, content,
            tags=flags.get("tags", ""),
            project=flags.get("project"),
            priority=int(flags.get("priority", 0)),
            related_to=flags.get("related"),
            expires_at=flags.get("expires"),
            source=flags.get("source", "manual"),
            topic_key=flags.get("key"),
            derived_from=flags.get("derived-from"),
            citations=flags.get("citations"),
            reasoning=flags.get("reasoning"),
            wing=flags.get("wing"),
            room=flags.get("room"),
            tier=flags.get("tier"),
            agent_id=flags.get("agent-id"),
            is_pinned=bool(flags.get("pin")),
        )

    elif cmd == "search" and len(sys.argv) >= 3:
        from .modes import get_mode_config
        from .display import show_token_economics, estimate_tokens

        flags, query_parts = parse_flags(sys.argv, 2)
        query = " ".join(query_parts)

        # Check for JSON output mode
        json_output = flags.get("json", False)

        # Determine search mode
        search_mode = "hybrid"  # default
        if flags.get("semantic"):
            search_mode = "semantic"
        elif flags.get("keyword"):
            search_mode = "keyword"

        # Temporal constraints
        since = flags.get("since")
        until = flags.get("until")

        # Recency boost control
        apply_recency = not flags.get("no-recency", False)

        # Reasoning boost control (ReasoningBank feature)
        apply_reasoning = not flags.get("no-reasoning-boost", False)

        # Reasoning bank filter: show only memories with confirmed predictions
        reasoning_bank_mode = flags.get("reasoning-bank", False)

        # Metadata pre-filters
        project_filter = flags.get("project")
        tags_filter = flags.get("tags")
        wing_filter = flags.get("wing")
        room_filter = flags.get("room")

        # Load passport credential for access control (from --passport flag or MEMORY_PASSPORT env var)
        passport_cred = None
        passport_path = flags.get("passport") or os.environ.get("MEMORY_PASSPORT")
        if passport_path:
            try:
                with open(passport_path, 'r') as f:
                    passport_cred = json.load(f)
                logger.debug(f"Loaded passport credential from {passport_path}")
            except FileNotFoundError:
                logger.warning(f"Passport file not found: {passport_path}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in passport file: {e}")

        rows, search_id, temporal_range = search_memories(
            query, mode=search_mode, since=since, until=until, apply_recency_boost=apply_recency,
            reasoning_boost=apply_reasoning, project=project_filter, tags=tags_filter,
            wing=wing_filter, room=room_filter, passport_credential=passport_cred
        )

        # Apply PPR boost if requested (Upgrade 3)
        if flags.get("ppr") and rows:
            from .ppr import ppr_boost_search_results
            # Convert rows to list of dicts with scores
            results_with_scores = []
            for i, r in enumerate(rows):
                result_dict = dict(r) if hasattr(r, 'keys') else r
                # Use inverse rank as score (1.0 for top result, decreasing)
                result_dict['score'] = 1.0 / (i + 1)
                results_with_scores.append(result_dict)

            # Apply PPR boost
            ppr_weight = float(flags.get("ppr-weight", "0.3"))
            boosted = ppr_boost_search_results(results_with_scores, ppr_weight=ppr_weight)
            rows = boosted
            if not json_output:
                print(f"🔗 PPR boost applied (weight: {ppr_weight:.1f})\n")

        # Apply reasoning bank filter if requested
        if reasoning_bank_mode and rows:
            from .reasoning import compute_reasoning_score
            conn = get_db()
            filtered_rows = []
            for r in rows:
                score, details = compute_reasoning_score(conn, r['id'])
                # Only include memories with confirmed predictions (score > 0.5)
                if details['confirmed'] > 0:
                    filtered_rows.append(r)
            conn.close()
            original_count = len(rows)
            rows = filtered_rows
            if original_count > len(rows) and not json_output:
                print(f"🧠 ReasoningBank filter: {len(rows)}/{original_count} memories have confirmed predictions\n")

        # Show temporal filter info if applied
        if temporal_range and not json_output:
            start, end = temporal_range
            print(f"📅 Filtered to: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}\n")

        # JSON output mode
        if json_output:
            output = []
            for r in rows:
                output.append(dict(r))
            print(json.dumps(output, ensure_ascii=False))
        elif rows:
            full_mode = flags.get("full", False)
            show_tokens = flags.get("tokens", False)  # --tokens flag to show individual estimates

            # Token budget: truncate results to fit within budget (claude-mem Steal #2)
            budget = flags.get("budget")
            if budget:
                try:
                    budget_limit = int(budget)
                    total_tokens = 0
                    truncated_rows = []
                    for r in rows:
                        tokens = estimate_tokens(r['content'])
                        if total_tokens + tokens <= budget_limit:
                            truncated_rows.append(r)
                            total_tokens += tokens
                        else:
                            # Stop when budget exceeded
                            break

                    if len(truncated_rows) < len(rows):
                        print(f"⚠️  Token budget: showing {len(truncated_rows)}/{len(rows)} results (fits in ~{total_tokens}/{budget_limit} tokens)\n")
                    rows = truncated_rows
                except ValueError:
                    pass  # Invalid budget, ignore

            for r in rows:
                if full_mode:
                    print(format_row(r))
                else:
                    print(format_row_compact(r, show_tokens=show_tokens))
                # Related in compact mode too
                if not full_mode:
                    for rel in get_related(r["id"]):
                        print(f"  -> #{rel['id']}: {rel['content'][:60]}")
                else:
                    for rel in get_related(r["id"]):
                        print(f"      -> #{rel['id']} ({rel['relation_type']}): {rel['content'][:80]}")

            # Show token economics summary (always enabled)
            show_token_economics(rows, compact=not full_mode)

            # Output search_id for feedback tracking (can be parsed by hooks)
            print(f"\n[search_id:{search_id}]")
        else:
            if not json_output:
                print("No memories found.")

    elif cmd == "get" and len(sys.argv) >= 3:
        flags, args = parse_flags(sys.argv, 2)
        mem_id = int(args[0]) if args else int(sys.argv[2])
        json_output = flags.get("json", False)

        if json_output:
            mem = get_memory(mem_id)
            if mem:
                print(json.dumps(dict(mem), ensure_ascii=False))
            else:
                print(json.dumps({"error": "Memory not found"}, ensure_ascii=False))
                sys.exit(1)
        else:
            print_memory_full(mem_id)

    # Legacy passport <memory_id> command removed - use 'get <memory_id>' for memory details
    # or 'passport --agent-id <id>' for W3C agent credentials

    elif cmd == "list":
        flags, _ = parse_flags(sys.argv, 2)
        json_output = flags.get("json", False)
        rows = list_memories(
            category=flags.get("category"),
            project=flags.get("project"),
            tag=flags.get("tag"),
            stale_only="stale" in sys.argv,
            expired_only="--expired" in sys.argv,
            sort_by_proof="--proven" in sys.argv,
            wing=flags.get("wing"),
            room=flags.get("room"),
        )
        if json_output:
            output = [dict(r) for r in rows]
            print(json.dumps(output, ensure_ascii=False))
        else:
            for r in rows:
                print(format_row(r))
            print(f"\n({len(rows)} memories)")

    elif cmd == "update" and len(sys.argv) >= 4:
        update_memory(int(sys.argv[2]), " ".join(sys.argv[3:]))

    elif cmd == "delete" and len(sys.argv) >= 3:
        delete_memory(int(sys.argv[2]))

    elif cmd == "erase" and len(sys.argv) >= 3:
        from .gdpr import erase_memory
        flags, args = parse_flags(sys.argv, 2)
        memory_id = int(args[0]) if args else int(sys.argv[2])
        reason = flags.get("reason", "gdpr_erasure_request")
        actor = flags.get("actor", "user")
        ok = erase_memory(memory_id, actor=actor, reason=reason)
        if ok:
            print(f"Memory #{memory_id} erased. Audit trail preserved.")
        else:
            print(f"Memory #{memory_id} not found or already inactive.", file=sys.stderr)
            sys.exit(1)

    elif cmd == "audit":
        from .gdpr import export_audit_log, audit_stats
        flags, _ = parse_flags(sys.argv, 2)
        since = flags.get("since")
        event_type = flags.get("type")
        limit = int(flags.get("limit", 50))

        entries = export_audit_log(since=since, event_type=event_type, limit=limit)
        if not entries:
            print("No audit log entries found.")
        else:
            for e in entries:
                mem_id = f"mem#{e['memory_id']}" if e['memory_id'] else "N/A"
                reason = e['reason'] or ''
                print(f"[{e['created_at']}] {e['event_type']:12} {mem_id:8} by {e['actor']:10} — {reason}")
            stats = audit_stats()
            print(f"\nTotal: {stats['total_entries']} entries | Oldest: {stats['oldest_entry']}")

    elif cmd == "pin" and len(sys.argv) >= 3:
        mem_id = int(sys.argv[2])
        conn = get_db()
        conn.execute("UPDATE memories SET is_pinned = 1 WHERE id = ?", (mem_id,))
        conn.commit()
        conn.close()
        print(f"📌 Memory #{mem_id} pinned (immune to decay/GC)")

    elif cmd == "unpin" and len(sys.argv) >= 3:
        mem_id = int(sys.argv[2])
        conn = get_db()
        conn.execute("UPDATE memories SET is_pinned = 0 WHERE id = ?", (mem_id,))
        conn.commit()
        conn.close()
        print(f"📌 Memory #{mem_id} unpinned")

    elif cmd == "tag" and len(sys.argv) >= 4:
        tag_memory(int(sys.argv[2]), sys.argv[3])

    elif cmd == "relate" and len(sys.argv) >= 4:
        rel_type = sys.argv[4] if len(sys.argv) > 4 else "related"
        relate_memories(int(sys.argv[2]), int(sys.argv[3]), rel_type)

    elif cmd == "conflicts":
        conflicts = find_conflicts()
        if conflicts:
            print(f"Potential conflicts ({len(conflicts)} found):\n")
            for c in conflicts:
                suggest = "MERGE" if c["score"] > 0.70 else "REVIEW"
                print(f"  #{c['id1']} vs #{c['id2']} ({c['score']:.0%} similar) [{c['project']}/{c['category']}]")
                print(f"    A: {c['content1'][:80]}...")
                print(f"    B: {c['content2'][:80]}...")
                print(f"    Suggest: {suggest} (memory-tool merge {c['id1']} {c['id2']})\n")
        else:
            print("No conflicts found.")

    elif cmd == "merge" and len(sys.argv) >= 4:
        merge_memories(int(sys.argv[2]), int(sys.argv[3]))

    elif cmd == "supersede" and len(sys.argv) >= 4:
        supersede_memory(int(sys.argv[2]), int(sys.argv[3]))

    elif cmd == "pending":
        rows = list_memories(category="pending")
        for r in rows:
            print(format_row(r))
        print(f"\n({len(rows)} pending items)")

    elif cmd == "projects":
        conn = get_db()
        projects = conn.execute("""
            SELECT project, COUNT(*) as count,
                   GROUP_CONCAT(DISTINCT category) as categories,
                   SUM(CASE WHEN category='pending' THEN 1 ELSE 0 END) as pending,
                   SUM(CASE WHEN stale=1 THEN 1 ELSE 0 END) as stale
            FROM memories WHERE active = 1 AND project IS NOT NULL
            GROUP BY project ORDER BY count DESC
        """).fetchall()
        conn.close()
        for p in projects:
            print(f"  {p['project']}: {p['count']} memories, {p['pending']} pending, {p['stale']} stale ({p['categories']})")

    elif cmd == "topics":
        export_topics()
        print("Exported topic files")

    elif cmd == "export":
        flags, _ = parse_flags(sys.argv, 2)
        project = flags.get("project") or detect_project()

        # Try phase-aware export first, fallback to classic export on error
        try:
            export_memory_md_phased(project)
            focus = f" (focused on {project})" if project else ""
            print(f"Exported to {MEMORY_MD_PATH}{focus} (phase-aware)")
        except Exception as e:
            logger.warning(f"Phase-aware export failed: {e}, falling back to classic export")
            export_memory_md(project)
            focus = f" (focused on {project})" if project else ""
            print(f"Exported to {MEMORY_MD_PATH}{focus}")

    elif cmd == "stats":
        flags, _ = parse_flags(sys.argv, 2)
        json_output = flags.get("json", False)

        conn = get_db()
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) as active,
                   SUM(CASE WHEN stale = 1 AND active = 1 THEN 1 ELSE 0 END) as stale,
                   SUM(CASE WHEN expires_at IS NOT NULL AND expires_at < datetime('now') AND active = 1 THEN 1 ELSE 0 END) as expired,
                   SUM(CASE WHEN is_pinned = 1 AND active = 1 THEN 1 ELSE 0 END) as pinned,
                   COUNT(DISTINCT project) as projects,
                   COUNT(DISTINCT category) as categories,
                   SUM(access_count) as total_accesses
            FROM memories
        """).fetchone()
        cats = conn.execute("""
            SELECT category, COUNT(*) as count, SUM(access_count) as accesses
            FROM memories WHERE active = 1 GROUP BY category ORDER BY count DESC
        """).fetchall()
        relations = conn.execute("SELECT COUNT(*) as c FROM memory_relations").fetchone()["c"]
        snapshots = conn.execute("SELECT COUNT(*) as c FROM session_snapshots").fetchone()["c"]
        sources = conn.execute("""
            SELECT source, COUNT(*) as c FROM memories WHERE active = 1 GROUP BY source ORDER BY c DESC
        """).fetchall()
        topic_keys = conn.execute("SELECT COUNT(*) as c FROM memories WHERE topic_key IS NOT NULL AND active = 1").fetchone()["c"]
        # Backup info
        backup_count = len(list(BACKUP_DIR.glob("memories_*.db"))) if BACKUP_DIR.exists() else 0

        # Vector index status
        vec_indexed = 0
        if has_vec_support():
            try:
                vec_indexed = conn.execute("SELECT COUNT(*) as c FROM memory_vec").fetchone()["c"]
            except sqlite3.OperationalError:
                pass  # Table doesn't exist yet

        # Graph stats
        g_stats = graph_stats()

        # Tier breakdown
        tiers = conn.execute("""
            SELECT tier, COUNT(*) as count, SUM(access_count) as accesses
            FROM memories WHERE active = 1 GROUP BY tier ORDER BY
            CASE tier
                WHEN 'semantic' THEN 1
                WHEN 'episodic' THEN 2
                WHEN 'working' THEN 3
            END
        """).fetchall()

        # Corrections status
        corrections_total = 0
        corrections_pending = 0
        try:
            corrections_total = conn.execute("SELECT COUNT(*) as c FROM corrections").fetchone()["c"]
            corrections_pending = conn.execute("SELECT COUNT(*) as c FROM corrections WHERE status = 'pending'").fetchone()["c"]
        except sqlite3.OperationalError:
            pass  # corrections table might not exist in older versions

        conn.close()

        # Build JSON object if needed
        if json_output:
            output = {
                "total": stats['total'],
                "active": stats['active'],
                "stale": stats['stale'],
                "expired": stats['expired'] or 0,
                "pinned": stats['pinned'] or 0,
                "projects": stats['projects'],
                "categories": stats['categories'],
                "total_accesses": stats['total_accesses'] or 0,
                "relations": relations,
                "snapshots": snapshots,
                "backups": backup_count,
                "topic_keys": topic_keys,
                "vector_indexed": vec_indexed,
                "vector_available": has_vec_support(),
                "graph": {
                    "entities": g_stats['entities'],
                    "relationships": g_stats['relationships'],
                    "facts": g_stats['facts'],
                    "memory_links": g_stats['memory_links']
                },
                "by_category": [dict(c) for c in cats],
                "by_tier": [dict(t) for t in tiers],
                "by_source": [dict(s) for s in sources],
                "corrections": {
                    "total": corrections_total,
                    "pending": corrections_pending
                }
            }

            # Add optional stats if available
            try:
                from .context_budget import budget_stats
                budget_info = budget_stats()
                if budget_info['total_memories'] > 0:
                    output['context_budget'] = budget_info
            except Exception:
                pass

            try:
                from .procedures import procedure_stats
                proc_stats = procedure_stats()
                if proc_stats['total_procedures'] > 0:
                    output['procedural'] = proc_stats
            except Exception:
                pass

            try:
                from .feedback import get_search_quality_stats
                search_stats = get_search_quality_stats()
                if search_stats['hit_rate_all']['searches'] > 0:
                    output['search_quality'] = {
                        "hit_rate_7d": search_stats['hit_rate_7d'],
                        "hit_rate_30d": search_stats['hit_rate_30d'],
                        "hit_rate_all": search_stats['hit_rate_all']
                    }
            except Exception:
                pass

            print(json.dumps(output, ensure_ascii=False))
        else:
            print(f"Memories: {stats['total']} total ({stats['active']} active, {stats['stale']} stale, {stats['expired'] or 0} expired, {stats['pinned'] or 0} pinned)")
            print(f"Projects: {stats['projects']} | Categories: {stats['categories']}")
            print(f"Relations: {relations} | Snapshots: {snapshots} | Backups: {backup_count}")
            print(f"Topic keys: {topic_keys}")
            print(f"Total accesses: {stats['total_accesses'] or 0}")
            print(f"Graph: {g_stats['entities']} entities, {g_stats['relationships']} relationships, {g_stats['facts']} facts, {g_stats['memory_links']} memory links")

            # Vector search status
            if has_vec_support():
                vec_pct = (vec_indexed / stats['active'] * 100) if stats['active'] > 0 else 0
                print(f"Vector index: {vec_indexed}/{stats['active']} embeddings ({vec_pct:.0f}%)")
            else:
                print("Vector index: Not available (install sqlite-vec, onnxruntime, tokenizers, numpy)")
            # Bridge status
            sync_state = load_sync_state()
            if sync_state.get('last_sync'):
                last_sync = sync_state['last_sync'][:16]
                openclaw_files = len(list(OPENCLAW_MEMORY_DIR.glob("*.md"))) if OPENCLAW_MEMORY_DIR.exists() else 0
                print(f"Bridge: last sync {last_sync}, {openclaw_files} OpenClaw files")
            else:
                print("Bridge: never synced (run 'memory-tool sync')")

            if corrections_total:
                print(f"Corrections: {corrections_total} total ({corrections_pending} pending)")

            print("\nBy category:")
            for c in cats:
                print(f"  {c['category']}: {c['count']} (accessed {c['accesses'] or 0}x)")

            if tiers:
                print("\nBy tier:")
                for t in tiers:
                    print(f"  {t['tier']}: {t['count']} (accessed {t['accesses'] or 0}x)")

            print("\nBy source:")
            for s in sources:
                print(f"  {s['source']}: {s['c']}")

            # Context budget stats (Upgrade 5)
            try:
                from .context_budget import budget_stats
                budget_info = budget_stats()
                if budget_info['total_memories'] > 0:
                    print(f"\nContext Budget:")
                    print(f"  Total tokens: ~{budget_info['total_tokens']:,}")
                    print(f"  Avg tokens/memory: ~{budget_info['avg_tokens']:.1f}")
                    print(f"  Range: {budget_info['min_tokens']} - {budget_info['max_tokens']} tokens")
            except Exception:
                pass  # context_budget module might not be available

            # Procedural memory stats (Upgrade 4)
            try:
                from .procedures import procedure_stats
                proc_stats = procedure_stats()
                if proc_stats['total_procedures'] > 0:
                    print(f"\nProcedural Memory:")
                    print(f"  Total procedures: {proc_stats['total_procedures']}")
                    print(f"  Overall success rate: {proc_stats['overall_success_rate']:.1f}%")
            except Exception:
                pass  # procedures module might not be available

            # Search quality summary
            try:
                from .feedback import get_search_quality_stats
                search_stats = get_search_quality_stats()
                if search_stats['hit_rate_all']['searches'] > 0:
                    print(f"\nSearch Quality:")
                    print(f"  Hit rate (7d): {search_stats['hit_rate_7d']['rate']:.0%} ({search_stats['hit_rate_7d']['searches']} searches)")
                    print(f"  Hit rate (all): {search_stats['hit_rate_all']['rate']:.0%} ({search_stats['hit_rate_all']['searches']} searches)")
            except Exception:
                pass  # Feedback module might not be available

    elif cmd == "tiers":
        conn = get_db()
        tier_stats = conn.execute("""
            SELECT tier, COUNT(*) as count, SUM(access_count) as accesses
            FROM memories WHERE active = 1 GROUP BY tier ORDER BY
            CASE tier
                WHEN 'semantic' THEN 1
                WHEN 'episodic' THEN 2
                WHEN 'working' THEN 3
            END
        """).fetchall()
        conn.close()

        print("Memory Tiers:")
        total = sum(t['count'] for t in tier_stats)
        for t in tier_stats:
            pct = (t['count'] / total * 100) if total > 0 else 0
            accesses = t['accesses'] or 0
            print(f"  {t['tier']}: {t['count']} ({pct:.0f}%) — {accesses} accesses")
        print(f"\nTotal: {total} active memories")
        print("\nTier meanings:")
        print("  semantic: Long-term knowledge (access_count >= 5, age > 7d)")
        print("  episodic: Recent memories (default)")
        print("  working: Short-term (expires within 24h)")

    elif cmd == "promote" and len(sys.argv) >= 3:
        from .tiers import promote_memory_to_tier
        try:
            mem_id = int(sys.argv[2])
            target_tier = sys.argv[3] if len(sys.argv) >= 4 else 'semantic'
            conn = get_db()
            promote_memory_to_tier(conn, mem_id, target_tier)
            conn.close()
            print(f"✓ Promoted memory #{mem_id} to {target_tier}")
        except ValueError as e:
            print(f"Error: {e}")

    elif cmd == "demote" and len(sys.argv) >= 3:
        from .tiers import demote_memory_to_tier
        try:
            mem_id = int(sys.argv[2])
            target_tier = sys.argv[3] if len(sys.argv) >= 4 else 'episodic'
            conn = get_db()
            demote_memory_to_tier(conn, mem_id, target_tier)
            conn.close()
            print(f"✓ Demoted memory #{mem_id} to {target_tier}")
        except ValueError as e:
            print(f"Error: {e}")

    elif cmd == "stale":
        rows = get_stale()
        if rows:
            for r in rows:
                print(format_row(r))
            print(f"\n({len(rows)} stale items)")
            print("Actions: 'memory-tool delete <id>' or 'memory-tool update <id> ...'")
        else:
            print("No stale memories.")

    elif cmd == "decay":
        run_decay()

    elif cmd == "passive-verify":
        from .memory_ops import passive_verify
        dry_run = "--dry-run" in sys.argv
        result = passive_verify(dry_run=dry_run)
        if result['dry_run']:
            print(f"DRY RUN: Would promote {len(result['promoted'])} memories")
        else:
            print(f"Passive verification complete:")
            print(f"  Promoted: {len(result['promoted'])} memories (search-verified tag + priority boost)")
            print(f"  Already verified: {len(result['already_verified'])}")
            print(f"  Checked: {result['checked']} candidates (appeared in search 3+ distinct days)")
        if result['promoted']:
            print(f"  IDs: {', '.join(str(i) for i in result['promoted'][:10])}" +
                  (f"... (+{len(result['promoted'])-10} more)" if len(result['promoted']) > 10 else ""))

    elif cmd == "consolidate":
        print("🧠 Running memory consolidation...")
        conn = get_db()
        results = consolidate_memories(conn)
        conn.close()
        print(f"\nConsolidation complete:")
        print(f"  📦 Merged: {results['merged']} near-duplicates")
        print(f"  💡 Insights: {results['insights']} patterns discovered")
        print(f"  🔗 Connections: {results['connections']} strengthened")
        print(f"  🗑️  Pruned: {results['pruned']} low-value memories")
        total = sum(results.values())
        if total == 0:
            print("\n  Memory is clean — nothing to consolidate. 🧘")

    elif cmd == "snapshot" and len(sys.argv) >= 3:
        flags, content_parts = parse_flags(sys.argv, 2)
        summary = " ".join(content_parts) if content_parts else " ".join(sys.argv[2:])
        project = flags.get("project") or detect_project()
        save_snapshot(summary, project)
        export_memory_md(project)

    elif cmd == "auto-snapshot":
        auto_snapshot()

    elif cmd == "snapshots":
        flags, _ = parse_flags(sys.argv, 2)
        limit = int(flags.get("limit", 5))
        snaps = get_snapshots(limit)
        for s in snaps:
            proj = f" [{s['project']}]" if s["project"] else ""
            files = f"\n    Files: {s['files_touched'][:150]}" if s["files_touched"] else ""
            print(f"  [{s['created_at'][:16]}]{proj} {s['summary']}{files}")
        print(f"\n({len(snaps)} snapshots)")

    elif cmd == "detect-project":
        print(detect_project() or "Unknown project")

    elif cmd == "gc":
        garbage_collect(int(sys.argv[2]) if len(sys.argv) >= 3 else 180)

    elif cmd == "log-error" and len(sys.argv) >= 4:
        command = sys.argv[2]
        error = " ".join(sys.argv[3:])
        flags, _ = parse_flags(sys.argv, 4)
        log_error(command, error, project=flags.get("project"))

    elif cmd == "import-md" and len(sys.argv) >= 3:
        import_session_md(sys.argv[2])

    elif cmd == "backup":
        backup_db()

    elif cmd == "restore" and len(sys.argv) >= 3:
        restore_db(sys.argv[2])

    elif cmd == "reindex":
        reindex_embeddings()

    elif cmd == "graph":
        # Graph intelligence subcommands
        if len(sys.argv) < 3:
            # Show graph summary
            g_stats = graph_stats()
            print(f"Graph Intelligence Summary:")
            print(f"  Entities: {g_stats['entities']}")
            print(f"  Relationships: {g_stats['relationships']}")
            print(f"  Facts: {g_stats['facts']}")
            print(f"  Memory links: {g_stats['memory_links']}")
            if g_stats['by_type']:
                print(f"\n  By type:")
                for t in g_stats['by_type']:
                    print(f"    {t['type']}: {t['count']}")
        else:
            subcmd = sys.argv[2]

            if subcmd == "add" and len(sys.argv) >= 5:
                entity_type = sys.argv[3]
                name = sys.argv[4]
                summary = " ".join(sys.argv[5:]) if len(sys.argv) > 5 else ""
                entity_id = graph_add_entity(name, entity_type, summary)
                print(f"Added entity #{entity_id}: {name} ({entity_type})")

            elif subcmd == "rel" and len(sys.argv) >= 6:
                from_name = sys.argv[3]
                relation_type = sys.argv[4]
                to_name = sys.argv[5]
                note = " ".join(sys.argv[6:]) if len(sys.argv) > 6 else ""
                success = graph_add_relationship(from_name, to_name, relation_type, note)
                if success:
                    print(f"Added relationship: {from_name} --{relation_type}--> {to_name}")
                else:
                    print("Failed to add relationship")

            elif subcmd == "fact" and len(sys.argv) >= 6:
                entity_name = sys.argv[3]
                key = sys.argv[4]
                value = " ".join(sys.argv[5:])
                success = graph_set_fact(entity_name, key, value)
                if success:
                    print(f"Set fact: {entity_name}.{key} = {value}")
                else:
                    print("Failed to set fact")

            elif subcmd == "get" and len(sys.argv) >= 4:
                name = sys.argv[3]
                entity = graph_get_entity(name)
                if entity:
                    e = entity['entity']
                    print(f"\n{e['name']} ({e['type']}) - Importance: {e['importance']}")
                    if e['summary']:
                        print(f"  Summary: {e['summary']}")
                    print(f"  Created: {e['created_at'][:16]} | Updated: {e['updated_at'][:16]}")

                    if entity['facts']:
                        print(f"\n  Facts ({len(entity['facts'])}):")
                        for f in entity['facts']:
                            conf = f" (confidence: {f['confidence']})" if f['confidence'] < 1.0 else ""
                            print(f"    {f['key']}: {f['value']}{conf}")

                    if entity['outgoing']:
                        print(f"\n  Relationships (outgoing):")
                        for r in entity['outgoing']:
                            note = f" - {r['note']}" if r['note'] else ""
                            print(f"    --{r['relation_type']}--> {r['to_name']} ({r['to_type']}){note}")

                    if entity['incoming']:
                        print(f"\n  Relationships (incoming):")
                        for r in entity['incoming']:
                            note = f" - {r['note']}" if r['note'] else ""
                            print(f"    <--{r['relation_type']}-- {r['from_name']} ({r['from_type']}){note}")

                    if entity['memories']:
                        print(f"\n  Linked memories ({len(entity['memories'])}):")
                        for m in entity['memories']:
                            print(f"    #{m['id']} [{m['category']}] {m['content'][:80]}")
                else:
                    print(f"Entity not found: {name}")

            elif subcmd == "list":
                entity_type = sys.argv[3] if len(sys.argv) >= 4 else None
                entities = graph_list_entities(entity_type)
                if entities:
                    for e in entities:
                        summary = f" - {e['summary'][:60]}" if e['summary'] else ""
                        print(f"  {e['name']} ({e['type']}) [importance: {e['importance']}]{summary}")
                    print(f"\n({len(entities)} entities)")
                else:
                    print("No entities found")

            elif subcmd == "delete" and len(sys.argv) >= 4:
                name = sys.argv[3]
                success = graph_delete_entity(name)
                if success:
                    print(f"Deleted entity: {name}")
                else:
                    print(f"Entity not found: {name}")

            elif subcmd == "spread" and len(sys.argv) >= 4:
                name = sys.argv[3]
                depth = int(sys.argv[4]) if len(sys.argv) >= 5 else 2
                results = graph_spread(name, depth)
                if results:
                    print(f"Spreading activation from '{name}' (depth={depth}):")
                    for e in results:
                        print(f"  [{e['activation']:.2f}] {e['name']} ({e['type']}) - {e['summary'][:60] if e['summary'] else '(no summary)'}")
                    print(f"\n({len(results)} entities)")
                else:
                    print(f"No connected entities found or entity '{name}' doesn't exist")

            elif subcmd == "link" and len(sys.argv) >= 5:
                memory_id = int(sys.argv[3])
                entity_name = sys.argv[4]
                success = link_memory_to_entity(memory_id, entity_name)
                if success:
                    print(f"Linked memory #{memory_id} to entity '{entity_name}'")
                else:
                    print("Failed to link memory to entity")

            elif subcmd == "auto-link":
                total_links, total_mems = graph_auto_link_all()
                print(f"Auto-linked {total_links} connections across {total_mems} memories")

            elif subcmd == "import-openclaw":
                graph_import_openclaw()

            elif subcmd == "stats":
                g_stats = graph_stats()
                print(f"Graph Statistics:")
                print(f"  Entities: {g_stats['entities']}")
                print(f"  Relationships: {g_stats['relationships']}")
                print(f"  Facts: {g_stats['facts']}")
                print(f"  Memory links: {g_stats['memory_links']}")
                if g_stats['by_type']:
                    print(f"\n  Entities by type:")
                    for t in g_stats['by_type']:
                        print(f"    {t['type']}: {t['count']}")

            else:
                print(f"Unknown graph subcommand: {subcmd}")
                print("\nGraph commands:")
                print("  memory-tool graph                         # Show graph summary")
                print("  memory-tool graph add <type> <name> [summary]")
                print("  memory-tool graph rel <from> <rel_type> <to> [note]")
                print("  memory-tool graph fact <entity> <key> <value>")
                print("  memory-tool graph get <name>")
                print("  memory-tool graph list [type]")
                print("  memory-tool graph delete <name>")
                print("  memory-tool graph spread <name> [depth]")
                print("  memory-tool graph link <memory_id> <entity_name>")
                print("  memory-tool graph auto-link")
                print("  memory-tool graph import-openclaw")
                print("  memory-tool graph stats")

    elif cmd == "sync":
        sync_bidirectional()

    elif cmd == "sync-to":
        sync_to_openclaw()

    elif cmd == "sync-from":
        sync_from_openclaw()

    elif cmd == "run":
        # Run tracking subcommands
        if len(sys.argv) < 3:
            print("Usage: memory-tool run <subcommand>")
            print("Subcommands: start, step, complete, fail, list, show, cancel")
            sys.exit(1)
            
        subcmd = sys.argv[2]
        
        if subcmd == "start" and len(sys.argv) >= 4:
            task = sys.argv[3]
            flags, _ = parse_flags(sys.argv, 4)
            agent = flags.get("agent", "claw")
            project = flags.get("project")
            tags = flags.get("tags", "")
            
            run_id = start_run(task, agent, project, tags)
            print(f"Started run #{run_id}")
            
        elif subcmd == "step" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                step_description = " ".join(sys.argv[4:])
                success = add_run_step(run_id, step_description)
                if success:
                    print(f"Added step to run #{run_id}")
                else:
                    print(f"Run #{run_id} not found")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "complete" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                outcome = " ".join(sys.argv[4:])
                complete_run(run_id, outcome)
                print(f"Completed run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "fail" and len(sys.argv) >= 5:
            try:
                run_id = int(sys.argv[3])
                reason = " ".join(sys.argv[4:])
                fail_run(run_id, reason)
                print(f"Failed run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "cancel" and len(sys.argv) >= 4:
            try:
                run_id = int(sys.argv[3])
                cancel_run(run_id)
                print(f"Cancelled run #{run_id}")
            except ValueError:
                print("Invalid run ID")
                
        elif subcmd == "list":
            flags, _ = parse_flags(sys.argv, 3)
            status = flags.get("status")
            project = flags.get("project")
            limit = int(flags.get("limit", 10))
            
            runs = list_runs(status, project, limit)
            
            if runs:
                print(f"{'ID':<4} {'Task':<50} {'Agent':<8} {'Status':<12} {'Started':<16} {'Duration':<12}")
                print("-" * 108)
                for r in runs:
                    task_preview = r['task'][:47] + "..." if len(r['task']) > 50 else r['task']
                    duration = format_duration(r['started_at'], r['completed_at'])
                    started_short = r['started_at'][:16] if r['started_at'] else "unknown"
                    print(f"{r['id']:<4} {task_preview:<50} {r['agent']:<8} {r['status']:<12} {started_short:<16} {duration:<12}")
                print(f"\n({len(runs)} runs)")
            else:
                print("No runs found")
                
        elif subcmd == "show" and len(sys.argv) >= 4:
            try:
                run_id = int(sys.argv[3])
                run = show_run(run_id)
                if run:
                    print(f"\n=== Run #{run['id']} ===")
                    print(f"Task: {run['task']}")
                    print(f"Agent: {run['agent']}")
                    print(f"Status: {run['status']}")
                    print(f"Started: {run['started_at']}")
                    if run['completed_at']:
                        print(f"Completed: {run['completed_at']}")
                        duration = format_duration(run['started_at'], run['completed_at'])
                        print(f"Duration: {duration}")
                    elif run['status'] == 'running':
                        duration = format_duration(run['started_at'])
                        print(f"Running for: {duration}")
                    if run['project']:
                        print(f"Project: {run['project']}")
                    if run['tags']:
                        print(f"Tags: {run['tags']}")
                    if run['outcome']:
                        print(f"Outcome: {run['outcome']}")
                    
                    # Parse and display steps
                    try:
                        steps = json.loads(run['steps'])
                        if steps:
                            print(f"\nSteps ({len(steps)}):")
                            for i, step in enumerate(steps, 1):
                                print(f"  {i}. {step}")
                    except (json.JSONDecodeError, TypeError):
                        if run['steps'] and run['steps'] != '[]':
                            print(f"Steps: {run['steps']}")
                    print()
                else:
                    print(f"Run #{run_id} not found")
            except ValueError:
                print("Invalid run ID")
        else:
            print(f"Unknown run subcommand: {subcmd}")
            print("\nRun commands:")
            print("  memory-tool run start \"task description\" [--agent claw|claude] [--project X] [--tags x,y]")
            print("  memory-tool run step <id> \"step description\"")
            print("  memory-tool run complete <id> \"outcome summary\"")
            print("  memory-tool run fail <id> \"reason\"")
            print("  memory-tool run list [--status running|completed|failed] [--project X] [--limit 10]")
            print("  memory-tool run show <id>")
            print("  memory-tool run cancel <id>")

    elif cmd == "importance":
        show_importance_ranking()

    elif cmd == "retention":
        # Show memories sorted by retention (lowest first)
        conn = get_db()
        now = datetime.now()
        rows = conn.execute("""
            SELECT id, category, project, content, fsrs_stability, fsrs_difficulty,
                   last_accessed_at, updated_at, access_count, priority
            FROM memories WHERE active = 1
            ORDER BY priority DESC
        """).fetchall()

        scored = []
        for r in rows:
            stability = r["fsrs_stability"] or 1.0
            last_acc = r["last_accessed_at"] or r["updated_at"]
            try:
                last_dt = datetime.fromisoformat(last_acc.replace('Z', '+00:00')).replace(tzinfo=None)
                elapsed = (now - last_dt).total_seconds() / 86400
            except (ValueError, AttributeError):
                elapsed = 90  # Fallback for invalid date format
            retention = fsrs_retention(stability, elapsed)
            scored.append((retention, r))

        scored.sort(key=lambda x: x[0])

        print("Memory Retention Report (lowest first — needs reinforcement)")
        print("=" * 60)
        for retention, r in scored[:20]:
            bar = "█" * int(retention * 10) + "░" * (10 - int(retention * 10))
            content = r["content"][:60]
            cat = r["category"][:5]
            proj = f"[{r['project'][:8]}]" if r["project"] else ""
            print(f"  #{r['id']:>3} {bar} {retention*100:>3.0f}% [{cat}]{proj} {content}")

        print(f"\n{len(scored)} memories total. Showing bottom 20.")
        conn.close()

    elif cmd == "next":
        suggest_next()

    elif cmd == "focus" and len(sys.argv) >= 3:
        from .focus import cmd_focus
        flags, topic_parts = parse_flags(sys.argv, 2)
        topic = " ".join(topic_parts) if topic_parts else sys.argv[2]
        full = flags.get("full", False)
        cmd_focus(topic, full)

    elif cmd == "dream":
        cmd_dream()

    elif cmd == "reflect":
        # Add a Reflexion-style reflection
        if len(sys.argv) < 3:
            print("Usage: memory-tool reflect \"<task_summary>\" [--outcome success|partial|failure] [--worked \"...\"] [--failed \"...\"] [--next \"...\"] [--project X]")
            print("\nInteractive mode:")
            print("  memory-tool reflect \"Fixed nginx config\"")
            print("\nBatch mode:")
            print("  memory-tool reflect \"Fixed nginx config\" --outcome partial --worked \"Checking syntax first\" --failed \"Forgot to reload\" --next \"Always run nginx -t && systemctl reload nginx\"")
            sys.exit(1)

        flags, task_parts = parse_flags(sys.argv, 2)
        task_summary = " ".join(task_parts) if task_parts else sys.argv[2]

        # Check if flags provided (batch mode) or interactive
        if flags.get("outcome"):
            outcome = flags.get("outcome")
            worked = flags.get("worked", "")
            failed = flags.get("failed", "")
            next_time = flags.get("next", "")
            project = flags.get("project")

            if outcome not in ['success', 'partial', 'failure']:
                print("Error: --outcome must be success, partial, or failure")
                sys.exit(1)
        else:
            # Interactive mode
            print(f"\nReflecting on: {task_summary}")
            print("-" * 60)

            outcome = input("Outcome (success/partial/failure): ").strip().lower()
            while outcome not in ['success', 'partial', 'failure']:
                print("Please enter success, partial, or failure")
                outcome = input("Outcome: ").strip().lower()

            worked = input("What worked well: ").strip()
            failed = input("What failed or didn't work: ").strip()
            next_time = input("What to do differently next time: ").strip()
            project = input("Project (optional, press Enter to skip): ").strip() or None

        mem_id = add_reflection(
            task_summary=task_summary,
            outcome=outcome,
            worked=worked,
            failed=failed,
            next_time=next_time,
            project=project
        )

        print(f"\n✅ Reflection #{mem_id} stored for: {task_summary}")
        print(f"   Outcome: {outcome}")
        print(f"   Use 'memory-tool reflect-load \"similar task\"' before starting related tasks")

    elif cmd == "reflect-load":
        # Load relevant reflections before starting a task
        if len(sys.argv) < 3:
            print("Usage: memory-tool reflect-load \"<task_description>\"")
            print("\nExample:")
            print("  memory-tool reflect-load \"nginx configuration\"")
            sys.exit(1)

        flags, query_parts = parse_flags(sys.argv, 2)
        task_description = " ".join(query_parts) if query_parts else sys.argv[2]

        reflections = load_reflections(task_description, limit=3)

        if not reflections:
            print(f"No relevant reflections found for: {task_description}")
            print("This is a new type of task. Good luck!")
            sys.exit(0)

        print(f"\n📚 Past Reflections for: {task_description}")
        print("=" * 70)

        for i, ref in enumerate(reflections, 1):
            outcome_emoji = {
                'success': '✅',
                'partial': '⚠️',
                'failure': '❌'
            }.get(ref['outcome'], '❓')

            print(f"\n{i}. {outcome_emoji} {ref['task'][:60]}")
            print(f"   ID: #{ref['id']} | Created: {ref['created_at'][:10]}")

            if ref['worked']:
                print(f"   ✓ What worked: {ref['worked'][:200]}")
            if ref['failed']:
                print(f"   ✗ What failed: {ref['failed'][:200]}")
            if ref['next_time']:
                print(f"   → Next time: {ref['next_time'][:200]}")

        print("\n" + "=" * 70)
        print("Use these insights to avoid past mistakes and build on what worked!")

    elif cmd == "lessons":
        # Show all reflections grouped by task type
        grouped = list_reflections_by_task()

        if not grouped:
            print("No reflections stored yet.")
            print("Start building your knowledge base with: memory-tool reflect \"<task>\"")
            sys.exit(0)

        print("\n📖 Lessons Learned by Task Type")
        print("=" * 70)

        # Calculate stats per task type
        for task_type in sorted(grouped.keys()):
            reflections = grouped[task_type]
            total = len(reflections)
            successes = sum(1 for r in reflections if r['outcome'] == 'success')
            failures = sum(1 for r in reflections if r['outcome'] == 'failure')
            partials = sum(1 for r in reflections if r['outcome'] == 'partial')

            print(f"\n{task_type.upper()} ({total} reflections)")
            print(f"  Success: {successes} | Partial: {partials} | Failure: {failures}")
            print(f"  {'':>5} {'Outcome':>8} {'Task':<50} {'ID':>6} {'Date':<12}")
            print(f"  {'-' * 85}")

            # Show most recent 5 for each task type
            for ref in reflections[:5]:
                outcome_emoji = {
                    'success': '✅',
                    'partial': '⚠️',
                    'failure': '❌'
                }.get(ref['outcome'], '❓')

                task_short = ref['task'][:50] if len(ref['task']) <= 50 else ref['task'][:47] + '...'
                date_short = ref['created_at'][:10]
                access = f"({ref['access_count']}×)" if ref['access_count'] > 0 else ""

                print(f"  {access:>5} {outcome_emoji} {ref['outcome']:>7} {task_short:<50} #{ref['id']:>5} {date_short}")

            if len(reflections) > 5:
                print(f"  ... and {len(reflections) - 5} more. Use 'memory-tool reflect-load \"{task_type}\"' to see all")

        print("\n" + "=" * 70)

        # Show patterns (task types with high failure rates)
        print("\n🔍 Patterns:")
        for task_type in sorted(grouped.keys()):
            reflections = grouped[task_type]
            if len(reflections) >= 3:
                failures = sum(1 for r in reflections if r['outcome'] == 'failure')
                failure_rate = failures / len(reflections)
                if failure_rate >= 0.5:
                    print(f"  ⚠️  {task_type}: {failure_rate:.0%} failure rate ({failures}/{len(reflections)}) - needs attention!")

    elif cmd == "correct":
        if len(sys.argv) < 3:
            print("Usage: memory-tool correct \"<correction text>\"")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        conn = get_db()
        conn.execute(
            "INSERT INTO corrections (raw_text, correction) VALUES (?, ?)",
            (text, text)
        )
        conn.commit()
        conn.close()
        print(f"✅ Correction queued: {text}")
        print("Run 'memory-tool corrections' to review pending corrections.")

    elif cmd == "corrections":
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM corrections WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
        conn.close()

        if not rows:
            print("No pending corrections. 🧠")
            sys.exit(0)

        print(f"Pending Corrections ({len(rows)})")
        print("=" * 50)
        for r in rows:
            print(f"  #{r['id']} [{r['category']}] {r['correction']}")
            print(f"     Raw: \"{r['raw_text']}\"")
            print(f"     Queued: {r['created_at']}")
            print()

        print("To apply: memory-tool apply-correction <id>")
        print("To dismiss: memory-tool dismiss-correction <id>")

    elif cmd == "apply-correction":
        if len(sys.argv) < 3:
            print("Usage: memory-tool apply-correction <id>")
            sys.exit(1)
        cid = int(sys.argv[2])
        conn = get_db()
        row = conn.execute("SELECT * FROM corrections WHERE id = ?", (cid,)).fetchone()
        if not row:
            print(f"Correction #{cid} not found.")
            conn.close()
            sys.exit(1)

        # Add as a preference/learning memory
        category = row["category"]
        content = row["correction"]

        # Use existing add_memory function
        mem_id = add_memory(category, content, tags="correction,user-feedback")

        conn.execute(
            "UPDATE corrections SET status = 'applied', applied_at = datetime('now'), memory_id = ? WHERE id = ?",
            (mem_id, cid)
        )
        conn.commit()
        conn.close()
        print(f"✅ Correction #{cid} applied as memory #{mem_id}")

    elif cmd == "dismiss-correction":
        if len(sys.argv) < 3:
            print("Usage: memory-tool dismiss-correction <id>")
            sys.exit(1)
        cid = int(sys.argv[2])
        conn = get_db()
        conn.execute("UPDATE corrections SET status = 'dismissed' WHERE id = ?", (cid,))
        conn.commit()
        conn.close()
        print(f"Correction #{cid} dismissed.")

    elif cmd == "detect":
        if len(sys.argv) < 3:
            print("Usage: memory-tool detect \"<text to scan>\"")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        result = detect_correction(text)
        if result:
            print(f"✅ Correction detected!")
            print(f"  Type: {result['type']}")
            print(f"  Match: {result['full_match']}")
            print(f"  Extracted: {result['correction']}")

            # Auto-queue it
            conn = get_db()
            conn.execute(
                "INSERT INTO corrections (raw_text, correction, category) VALUES (?, ?, 'preference')",
                (text, result['correction'])
            )
            conn.commit()
            conn.close()
            print(f"  → Queued as pending correction")
        else:
            print("No correction pattern detected.")

    elif cmd == "capture-correction":
        text = " ".join(sys.argv[2:])
        if not text:
            print("Usage: memory-tool capture-correction <text>")
            sys.exit(1)
        cmd_capture_correction(text)

    elif cmd == "feedback" and len(sys.argv) >= 4 and sys.argv[2].isdigit():
        # memory-tool feedback <search_id> <used_id1,used_id2,...>
        search_id = int(sys.argv[2])
        used_ids = [int(x.strip()) for x in sys.argv[3].split(',') if x.strip()]
        from .feedback import log_search_feedback
        log_search_feedback(search_id, used_ids)
        print(f"Feedback logged for search #{search_id}: {len(used_ids)} memories marked as used")

    elif cmd == "search-quality":
        from .feedback import get_search_quality_stats
        stats = get_search_quality_stats()

        print("Search Quality Report")
        print("=" * 70)

        # Hit rates by period
        print("\nOverall Hit Rates:")
        print(f"  Last 7 days:  {stats['hit_rate_7d']['rate']:.1%} ({stats['hit_rate_7d']['searches']} searches)")
        print(f"  Last 30 days: {stats['hit_rate_30d']['rate']:.1%} ({stats['hit_rate_30d']['searches']} searches)")
        print(f"  All time:     {stats['hit_rate_all']['rate']:.1%} ({stats['hit_rate_all']['searches']} searches)")

        # Most helpful memories
        if stats['most_helpful']:
            print("\nMost Helpful Memories (high hit rate, 5+ retrievals):")
            for m in stats['most_helpful'][:5]:
                content_preview = m['content'][:60] + "..." if len(m['content']) > 60 else m['content']
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['hit_rate']:.0%} ({m['used_count']}/{m['retrieve_count']}) {content_preview}")

        # Least helpful memories
        if stats['least_helpful']:
            print("\nLeast Helpful Memories (low hit rate, 10+ retrievals):")
            for m in stats['least_helpful'][:5]:
                content_preview = m['content'][:60] + "..." if len(m['content']) > 60 else m['content']
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['hit_rate']:.0%} ({m['used_count']}/{m['retrieve_count']}) {content_preview}")

        # Search patterns
        if stats['search_patterns']:
            print("\nMost Common Queries:")
            for p in stats['search_patterns'][:5]:
                print(f"  '{p['query'][:40]}' - {p['search_count']} searches, {p['avg_hit_rate']:.0%} avg hit rate")

        # Failing queries
        if stats['failing_queries']:
            print("\nRecent Failing Queries (0% hit rate):")
            for q in stats['failing_queries'][:5]:
                print(f"  '{q['query'][:50]}' [{q['search_type']}] - {q['result_count']} results but none used")

    elif cmd == "feedback" and len(sys.argv) == 3 and sys.argv[2] == "suggestions":
        # memory-tool feedback suggestions
        from .feedback import get_improvement_suggestions
        suggestions = get_improvement_suggestions()

        print("Improvement Suggestions")
        print("=" * 70)

        # Deprecation candidates
        if suggestions['deprecation_candidates']:
            print("\n🗑️  Deprecation Candidates (retrieved often, never used):")
            for m in suggestions['deprecation_candidates']:
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['retrieved_count']} retrievals")
                print(f"      {m['content']}")
                print(f"      Suggest: memory-tool delete {m['id']}")
        else:
            print("\n✓ No deprecation candidates found")

        # Knowledge gaps
        if suggestions['knowledge_gaps']:
            print("\n🔍 Knowledge Gaps (queries with no results):")
            for g in suggestions['knowledge_gaps']:
                print(f"  '{g['query']}' - {g['attempt_count']} failed searches")
                print(f"      Suggest: Add memory covering this topic")
        else:
            print("\n✓ No knowledge gaps found")

        # Tag suggestions
        if suggestions['tag_suggestions']:
            print("\n🏷️  Tag Suggestions (commonly searched terms):")
            for t in suggestions['tag_suggestions']:
                print(f"  '{t['keyword']}' - {t['search_count']} searches")
        else:
            print("\n✓ No tag suggestions")

    elif cmd == "feedback" and len(sys.argv) == 3 and sys.argv[2] == "reset":
        # memory-tool feedback reset
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) as c FROM search_log").fetchone()['c']
        conn.execute("DELETE FROM search_log")
        conn.commit()
        conn.close()
        print(f"Cleared {count} search log entries")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] in ["good", "bad", "meh"]:
        # memory-tool feedback good/bad/meh "reason"
        rating = sys.argv[2]
        reason = sys.argv[3] if len(sys.argv) >= 4 else None

        # Get most recent memory
        conn = get_db()
        last_memory = conn.execute("""
            SELECT id FROM memories
            WHERE active = 1
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()
        linked_memory_id = last_memory['id'] if last_memory else None

        # Try to get session ID from metrics file
        session_id = None
        try:
            from pathlib import Path
            metrics_file = Path("/tmp/claude-session-metrics.json")
            if metrics_file.exists():
                data = json.loads(metrics_file.read_text())
                session_id = data.get('startedAt', None)
        except Exception:
            pass

        # Insert feedback
        conn.execute("""
            INSERT INTO feedback (rating, reason, session_id, linked_memory_id)
            VALUES (?, ?, ?, ?)
        """, (rating, reason, session_id, linked_memory_id))
        conn.commit()

        memory_info = f" (linked to memory #{linked_memory_id})" if linked_memory_id else ""
        print(f"✓ Feedback recorded: {rating}{memory_info}")
        if reason:
            print(f"  Reason: {reason}")
        conn.close()

    elif cmd == "feedback" and len(sys.argv) == 2:
        # memory-tool feedback (no args) - show recent feedback
        conn = get_db()
        rows = conn.execute("""
            SELECT f.id, f.rating, f.reason, f.created_at, f.linked_memory_id,
                   m.content as memory_content, m.category
            FROM feedback f
            LEFT JOIN memories m ON f.linked_memory_id = m.id
            ORDER BY f.created_at DESC
            LIMIT 10
        """).fetchall()
        conn.close()

        if rows:
            print("Recent Feedback")
            print("=" * 70)
            for r in rows:
                created = r['created_at'][:16] if r['created_at'] else "unknown"
                rating_emoji = "✓" if r['rating'] == "good" else "✗" if r['rating'] == "bad" else "~"
                print(f"\n{rating_emoji} #{r['id']} [{r['rating'].upper()}] {created}")
                if r['reason']:
                    print(f"  Reason: {r['reason']}")
                if r['linked_memory_id']:
                    content = r['memory_content'][:60] + "..." if r['memory_content'] and len(r['memory_content']) > 60 else r['memory_content']
                    print(f"  Memory #{r['linked_memory_id']} [{r['category']}]: {content}")
            print(f"\n({len(rows)} feedback entries)")
        else:
            print("No feedback recorded yet.")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "record-surface":
        # memory-tool feedback record-surface --chat-id "123" --memory-ids "1,2,3" --message "user message" --assistant-message "bot response"
        flags, _ = parse_flags(sys.argv, 3)

        chat_id = flags.get("chat-id", "")
        memory_ids_str = flags.get("memory-ids", "")
        message = flags.get("message", "")
        assistant_message = flags.get("assistant-message", "")

        if not chat_id or not memory_ids_str:
            print("Error: --chat-id and --memory-ids are required")
            sys.exit(1)

        conn = get_db()

        # Add last_assistant_message column if not present (migration)
        try:
            conn.execute("ALTER TABLE feedback_state ADD COLUMN last_assistant_message TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Auto-cleanup: delete rows older than 1 hour
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        conn.execute("DELETE FROM feedback_state WHERE surfaced_at < ?", (one_hour_ago,))

        # Insert or replace
        conn.execute("""
            INSERT OR REPLACE INTO feedback_state
            (chat_id, surfaced_ids, last_message, last_assistant_message, turns_since, surfaced_at)
            VALUES (?, ?, ?, ?, 0, datetime('now'))
        """, (chat_id, memory_ids_str, message, assistant_message))
        conn.commit()
        conn.close()
        print(f"✓ Recorded surface state for chat {chat_id}")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "get-state":
        # memory-tool feedback get-state --chat-id "123"
        flags, _ = parse_flags(sys.argv, 3)
        chat_id = flags.get("chat-id", "")

        if not chat_id:
            print("Error: --chat-id is required")
            sys.exit(1)

        conn = get_db()

        # Add last_assistant_message column if not present (migration)
        try:
            conn.execute("ALTER TABLE feedback_state ADD COLUMN last_assistant_message TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Column already exists

        row = conn.execute("""
            SELECT chat_id, surfaced_ids, last_message, last_assistant_message, turns_since
            FROM feedback_state WHERE chat_id = ?
        """, (chat_id,)).fetchone()
        conn.close()

        if row:
            result = {
                'chat_id': row['chat_id'],
                'surfaced_ids': [int(x.strip()) for x in row['surfaced_ids'].split(',') if x.strip()],
                'last_message': row['last_message'],
                'last_assistant_message': row['last_assistant_message'] if row['last_assistant_message'] else '',
                'turns_since': row['turns_since']
            }
            print(json.dumps(result))
        else:
            print("null")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "clear-state":
        # memory-tool feedback clear-state --chat-id "123"
        flags, _ = parse_flags(sys.argv, 3)
        chat_id = flags.get("chat-id", "")

        if not chat_id:
            print("Error: --chat-id is required")
            sys.exit(1)

        conn = get_db()
        conn.execute("DELETE FROM feedback_state WHERE chat_id = ?", (chat_id,))
        conn.commit()
        conn.close()
        print(f"✓ Cleared state for chat {chat_id}")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "log-usage":
        # memory-tool feedback log-usage --search-id 42 --used-ids "1,3,7"
        flags, _ = parse_flags(sys.argv, 3)
        search_id = flags.get("search-id", "")
        used_ids_str = flags.get("used-ids", "")

        if not search_id or not used_ids_str:
            print("Error: --search-id and --used-ids are required")
            sys.exit(1)

        conn = get_db()
        conn.execute("UPDATE search_log SET used_ids = ? WHERE id = ?", (used_ids_str, int(search_id)))

        # Also increment citation_count on each cited memory (deduplicated)
        cited_ids = list(set(int(x.strip()) for x in used_ids_str.split(',') if x.strip()))
        if cited_ids:
            placeholders = ','.join('?' * len(cited_ids))
            conn.execute(
                f"UPDATE memories SET citation_count = citation_count + 1 WHERE id IN ({placeholders})",
                cited_ids
            )

        conn.commit()
        conn.close()
        print(f"✓ Logged usage for search #{search_id}")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "last-search-id":
        # memory-tool feedback last-search-id
        conn = get_db()
        row = conn.execute("SELECT MAX(id) as last_id FROM search_log").fetchone()
        conn.close()
        if row and row['last_id']:
            print(row['last_id'])
        else:
            print("0")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "process":
        # memory-tool feedback process --current-message "..." --previous-message "..." --memory-ids "1,2,3" --turns-since-surface 2 --cited-ids "1,3" --assistant-message "..."
        flags, _ = parse_flags(sys.argv, 3)

        current_message = flags.get("current-message", "")
        previous_message = flags.get("previous-message", "")
        assistant_message = flags.get("assistant-message", "")
        memory_ids_str = flags.get("memory-ids", "")
        cited_ids_str = flags.get("cited-ids", "")
        turns_since_surface = int(flags.get("turns-since-surface", 0))

        if not current_message:
            print("Error: --current-message is required")
            sys.exit(1)

        memory_ids = [int(x.strip()) for x in memory_ids_str.split(',') if x.strip()]
        cited_ids = [int(x.strip()) for x in cited_ids_str.split(',') if x.strip()] if cited_ids_str else None

        from .implicit_feedback import process_turn_feedback
        results = process_turn_feedback(
            conn=None,
            current_message=current_message,
            previous_message=previous_message,
            surfaced_memory_ids=memory_ids,
            turns_since_surface=turns_since_surface,
            cited_memory_ids=cited_ids,
            previous_assistant_message=assistant_message if assistant_message else None
        )

        # Output as JSON for bot consumption
        print(json.dumps(results))

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "record-surface":
        # memory-tool feedback record-surface --chat-id "123" --memory-ids "1,2,3" --message "text" --assistant-message "..."
        flags, _ = parse_flags(sys.argv, 3)
        chat_id = flags.get("chat-id", "")
        memory_ids_str = flags.get("memory-ids", "")
        message = flags.get("message", "")
        assistant_message = flags.get("assistant-message", "")
        if not chat_id:
            print("Error: --chat-id required"); sys.exit(1)
        conn = get_db()
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback_state (
            chat_id TEXT PRIMARY KEY,
            surfaced_ids TEXT NOT NULL,
            last_message TEXT NOT NULL,
            turns_since INTEGER DEFAULT 0,
            surfaced_at TEXT NOT NULL
        )""")
        # Add last_assistant_message column if not present
        try:
            conn.execute("ALTER TABLE feedback_state ADD COLUMN last_assistant_message TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        # TTL cleanup: delete rows older than 1 hour
        conn.execute("DELETE FROM feedback_state WHERE surfaced_at < datetime('now', '-24 hours')")
        conn.execute("""INSERT OR REPLACE INTO feedback_state
            (chat_id, surfaced_ids, last_message, last_assistant_message, turns_since, surfaced_at)
            VALUES (?, ?, ?, ?, 0, datetime('now'))""", (chat_id, memory_ids_str, message, assistant_message))
        conn.commit(); conn.close()
        print("ok")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "get-state":
        # memory-tool feedback get-state --chat-id "123"
        flags, _ = parse_flags(sys.argv, 3)
        chat_id = flags.get("chat-id", "")
        if not chat_id:
            print("null"); sys.exit(0)
        conn = get_db()
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS feedback_state (chat_id TEXT PRIMARY KEY, surfaced_ids TEXT NOT NULL, last_message TEXT NOT NULL, turns_since INTEGER DEFAULT 0, surfaced_at TEXT NOT NULL)")
            # Add last_assistant_message column if not present
            try:
                conn.execute("ALTER TABLE feedback_state ADD COLUMN last_assistant_message TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            row = conn.execute("SELECT * FROM feedback_state WHERE chat_id = ? AND surfaced_at > datetime('now', '-24 hours')", (chat_id,)).fetchone()
        except Exception:
            row = None
        conn.close()
        if row:
            ids = [int(x.strip()) for x in row['surfaced_ids'].split(',') if x.strip()]
            last_asst = row['last_assistant_message'] if 'last_assistant_message' in row.keys() else ''
            print(json.dumps({"chat_id": row['chat_id'], "surfaced_ids": ids, "last_message": row['last_message'], "last_assistant_message": last_asst, "turns_since": row['turns_since']}))
        else:
            print("null")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "clear-state":
        # memory-tool feedback clear-state --chat-id "123"
        flags, _ = parse_flags(sys.argv, 3)
        chat_id = flags.get("chat-id", "")
        if chat_id:
            conn = get_db()
            try:
                conn.execute("DELETE FROM feedback_state WHERE chat_id = ?", (chat_id,))
                conn.commit()
            except Exception:
                pass
            conn.close()
        print("ok")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "log-usage":
        # memory-tool feedback log-usage --search-id 42 --used-ids "1,3,7"
        flags, _ = parse_flags(sys.argv, 3)
        search_id = flags.get("search-id", "")
        used_ids_str = flags.get("used-ids", "")
        if search_id and used_ids_str:
            conn = get_db()
            try:
                conn.execute("UPDATE search_log SET used_ids = ? WHERE id = ?", (used_ids_str, int(search_id)))

                # Also increment citation_count on each cited memory (deduplicated)
                cited_ids = list(set(int(x.strip()) for x in used_ids_str.split(',') if x.strip()))
                if cited_ids:
                    placeholders = ','.join('?' * len(cited_ids))
                    conn.execute(
                        f"UPDATE memories SET citation_count = citation_count + 1 WHERE id IN ({placeholders})",
                        cited_ids
                    )

                conn.commit()
            except Exception:
                pass
            conn.close()
        print("ok")

    elif cmd == "feedback" and len(sys.argv) >= 3 and sys.argv[2] == "last-search-id":
        # memory-tool feedback last-search-id
        conn = get_db()
        try:
            row = conn.execute("SELECT MAX(id) as last_id FROM search_log").fetchone()
            print(row['last_id'] if row and row['last_id'] else "0")
        except Exception:
            print("0")
        conn.close()

    elif cmd == "feedback" and len(sys.argv) == 3 and sys.argv[2] == "--stats":
        # memory-tool feedback --stats
        conn = get_db()
        stats = conn.execute("""
            SELECT
                rating,
                COUNT(*) as count,
                COUNT(DISTINCT session_id) as sessions,
                COUNT(DISTINCT linked_memory_id) as unique_memories
            FROM feedback
            GROUP BY rating
        """).fetchall()
        total = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()['c']
        conn.close()

        if total > 0:
            print("Feedback Statistics")
            print("=" * 70)
            for s in stats:
                pct = (s['count'] / total * 100) if total > 0 else 0
                print(f"  {s['rating'].upper()}: {s['count']} ({pct:.0f}%) — {s['sessions']} sessions, {s['unique_memories']} memories")
            print(f"\nTotal: {total} feedback entries")
        else:
            print("No feedback recorded yet.")

    elif cmd == "feedback-stats":
        # Alias for search-quality
        from .feedback import get_search_quality_stats
        stats = get_search_quality_stats()

        print("Search Quality Report")
        print("=" * 70)

        # Hit rates by period
        print("\nOverall Hit Rates:")
        print(f"  Last 7 days:  {stats['hit_rate_7d']['rate']:.1%} ({stats['hit_rate_7d']['searches']} searches)")
        print(f"  Last 30 days: {stats['hit_rate_30d']['rate']:.1%} ({stats['hit_rate_30d']['searches']} searches)")
        print(f"  All time:     {stats['hit_rate_all']['rate']:.1%} ({stats['hit_rate_all']['searches']} searches)")

        # Most helpful memories
        if stats['most_helpful']:
            print("\nMost Helpful Memories (high hit rate, 5+ retrievals):")
            for m in stats['most_helpful'][:5]:
                content_preview = m['content'][:60] + "..." if len(m['content']) > 60 else m['content']
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['hit_rate']:.0%} ({m['used_count']}/{m['retrieve_count']}) {content_preview}")

        # Least helpful memories
        if stats['least_helpful']:
            print("\nLeast Helpful Memories (low hit rate, 10+ retrievals):")
            for m in stats['least_helpful'][:5]:
                content_preview = m['content'][:60] + "..." if len(m['content']) > 60 else m['content']
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['hit_rate']:.0%} ({m['used_count']}/{m['retrieve_count']}) {content_preview}")

        # Search patterns
        if stats['search_patterns']:
            print("\nMost Common Queries:")
            for p in stats['search_patterns'][:5]:
                print(f"  '{p['query'][:40]}' - {p['search_count']} searches, {p['avg_hit_rate']:.0%} avg hit rate")

        # Failing queries
        if stats['failing_queries']:
            print("\nRecent Failing Queries (0% hit rate):")
            for q in stats['failing_queries'][:5]:
                print(f"  '{q['query'][:50]}' [{q['search_type']}] - {q['result_count']} results but none used")

    elif cmd == "gaps":
        # Show knowledge gaps - queries with no results
        from .feedback import get_improvement_suggestions
        suggestions = get_improvement_suggestions()

        print("Knowledge Gaps (searches with poor results)")
        print("=" * 70)

        if suggestions['knowledge_gaps']:
            print("\nQueries with zero results:")
            for g in suggestions['knowledge_gaps']:
                print(f"  '{g['query']}' - {g['attempt_count']} failed attempt(s)")
                print(f"    Type: {g['search_type']}")
                print(f"    Suggestion: Add memory covering this topic")
                print()
        else:
            print("\n✓ No knowledge gaps found")

        if suggestions['deprecation_candidates']:
            print("\nMemories retrieved often but never used (false positives):")
            for m in suggestions['deprecation_candidates'][:5]:
                print(f"  #{m['id']:>3} [{m['category']:<8}] {m['retrieved_count']} retrievals")
                print(f"      {m['content']}")
                print(f"      Suggestion: Consider deletion or rewrite")
                print()

    elif cmd == "hot":
        # Show most frequently accessed memories (immune to decay)
        conn = get_db()
        rows = conn.execute("""
            SELECT id, category, project, content, access_count, priority, imp_score
            FROM memories
            WHERE active = 1 AND access_count >= 5
            ORDER BY access_count DESC
            LIMIT 20
        """).fetchall()
        conn.close()

        if rows:
            print("Hot Memories (5+ accesses, immune to decay)")
            print("=" * 70)
            for r in rows:
                content = r['content'][:50] + "..." if len(r['content']) > 50 else r['content']
                cat = r['category'][:8]
                proj = f"[{r['project'][:10]}]" if r['project'] else ""
                print(f"  #{r['id']:>3} {r['access_count']:>3}x [{cat:<8}]{proj:<12} {content}")
            print(f"\n({len(rows)} hot memories)")
        else:
            print("No hot memories yet (need 5+ accesses)")

    elif cmd == "reasoning":
        # Show memories ranked by reasoning boost (ReasoningBank pattern)
        from .reasoning import list_memories_by_reasoning

        results = list_memories_by_reasoning()

        if results:
            print("Memories Ranked by Reasoning Boost (confirmed vs refuted predictions)")
            print("=" * 80)
            for mem_id, preview, confirmed, refuted, boost in results:
                status = f"✓{confirmed}" if confirmed > 0 else ""
                if refuted > 0:
                    status += f" ✗{refuted}" if status else f"✗{refuted}"
                if not status:
                    status = "-"
                boost_str = f"{boost:.2f}x"
                print(f"  #{mem_id:>3} {boost_str:>6} [{status:>6}] {preview}")
            print(f"\n({len(results)} memories with predictions)")
            print("\nLegend: ✓ = confirmed predictions, ✗ = refuted predictions")
            from .config import REASONING_BOOST_BASE, REASONING_BOOST_CAP
            print(f"Boost formula: {REASONING_BOOST_BASE}^confirmed / {REASONING_BOOST_BASE}^refuted, capped at {REASONING_BOOST_CAP}x")
        else:
            print("No memories with linked predictions yet")
            print("\nTo create predictions: memory-tool predict \"prediction text\" --based-on <memory_id>")

    elif cmd == "validate" and len(sys.argv) >= 3:
        from .validation import (
            find_drift_candidates,
            mark_validated,
            mark_refuted,
            get_unvalidated_semantic,
            detect_contradictions_in_tier,
            validation_report
        )

        action = sys.argv[2]
        conn = get_db()

        if action == "scan":
            flags, _ = parse_flags(sys.argv, 3)
            min_access = int(flags.get("min-access", 5))
            min_age_days = int(flags.get("min-age-days", 30))

            candidates = find_drift_candidates(conn, min_access, min_age_days)

            if candidates:
                print("⚠️  Drift Detection: High-Risk Memories Needing Validation")
                print("=" * 80)
                print("These memories are frequently accessed but may be wrong (reinforced lies).")
                print()

                for mem in candidates[:15]:  # Show top 15
                    risk = mem['drift_risk']
                    risk_level = "🔴 HIGH" if risk > 0.7 else "🟡 MED" if risk > 0.5 else "🟢 LOW"
                    content = mem['content'][:60] + "..." if len(mem['content']) > 60 else mem['content']

                    print(f"  #{mem['id']:>3} {risk_level} (risk={risk:.2f}) [{mem['tier']:<8}] {mem['access_count']:>2}x")
                    print(f"      {content}")
                    print(f"      Last validated: {mem['last_validated_at'] or 'NEVER'}")
                    print()

                print(f"Total: {len(candidates)} drift candidates")
                print("\nActions:")
                print("  memory-tool validate confirm <id> --notes 'verified from X'")
                print("  memory-tool validate refute <id> --notes 'this is wrong because Y'")
            else:
                print("✓ No drift candidates found. All memories look healthy.")

            conn.close()

        elif action == "confirm" and len(sys.argv) >= 4:
            memory_id = int(sys.argv[3])
            flags, _ = parse_flags(sys.argv, 4)
            notes = flags.get("notes", "")
            validator = flags.get("validator", "user")

            success = mark_validated(conn, memory_id, validator, "user", "confirmed", notes)
            if success:
                print(f"✓ Memory #{memory_id} validated as CONFIRMED")
            else:
                print(f"✗ Failed to validate memory #{memory_id}")

            conn.close()

        elif action == "refute" and len(sys.argv) >= 4:
            memory_id = int(sys.argv[3])
            flags, _ = parse_flags(sys.argv, 4)
            notes = flags.get("notes", "")
            validator = flags.get("validator", "user")

            success = mark_refuted(conn, memory_id, validator, notes)
            if success:
                print(f"✓ Memory #{memory_id} marked as REFUTED and demoted")
            else:
                print(f"✗ Failed to refute memory #{memory_id}")

            conn.close()

        elif action == "list-unvalidated":
            unvalidated = get_unvalidated_semantic(conn)

            if unvalidated:
                print("⚠️  Unvalidated Semantic Memories")
                print("=" * 80)
                print("These are 'proven knowledge' that has never been validated.")
                print()

                for mem in unvalidated[:20]:
                    content = mem['content'][:60] + "..." if len(mem['content']) > 60 else mem['content']
                    print(f"  #{mem['id']:>3} [{mem['category']:<10}] {mem['access_count']:>2}x accesses")
                    print(f"      {content}")
                    print()

                print(f"Total: {len(unvalidated)} unvalidated semantic memories")
            else:
                print("✓ All semantic memories have been validated")

            conn.close()

        elif action == "report":
            report = validation_report(conn)

            print("📊 Validation Report")
            print("=" * 80)
            print()

            # Validation counts
            if report['validation_counts']:
                print("Validation Results:")
                for result, count in report['validation_counts'].items():
                    print(f"  {result.capitalize()}: {count}")
                print()

            # Tier stats
            print("Validation Status by Tier:")
            for tier in ['semantic', 'episodic', 'working']:
                if tier in report['tier_stats']:
                    stats = report['tier_stats'][tier]
                    pct = stats['pct_validated']
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"  {tier.capitalize():<10} [{bar}] {pct:>5.1f}%  ({stats['validated']}/{stats['total']})")

            print()
            print(f"High-risk memories needing validation: {report['high_risk_count']}")
            print(f"Total validations performed: {report['total_validations']}")

            conn.close()

        else:
            print("Usage: memory-tool validate <action> [args]")
            print("Actions:")
            print("  scan                      - Show drift candidates with risk scores")
            print("  confirm <id> --notes X    - Mark memory as validated")
            print("  refute <id> --notes Y     - Mark memory as wrong and demote")
            print("  list-unvalidated          - Show unvalidated semantic memories")
            print("  report                    - Show validation statistics")

    elif cmd == "validate":
        # Show help if no action specified
        print("Usage: memory-tool validate <action> [args]")
        print("Actions:")
        print("  scan                      - Show drift candidates with risk scores")
        print("  confirm <id> --notes X    - Mark memory as validated")
        print("  refute <id> --notes Y     - Mark memory as wrong and demote")
        print("  list-unvalidated          - Show unvalidated semantic memories")
        print("  report                    - Show validation statistics")

    elif cmd == "tier-evolution":
        from .tier_evolution import run_tier_evolution

        if len(sys.argv) < 3:
            print("Usage: memory-tool tier-evolution <subcommand>")
            print("Subcommands:")
            print("  run    - Run tier evolution batch job")
            print("  stats  - Show tier distribution and evolution stats")
            sys.exit(1)

        subcmd = sys.argv[2]

        if subcmd == "run":
            conn = get_db()
            results = run_tier_evolution(conn)
            conn.commit()
            conn.close()

            print("Tier Evolution Complete:")
            print(f"  Expired: {results['expired']} Tier 0 memories (90+ days no access)")
            print(f"  Stability-demoted: {results['stability_demoted']} memories (failed stability gate)")
            print(f"  Tier 3 demoted: {results['tier3_demoted']} memories (failed Tier 3 requirements)")

        elif subcmd == "stats":
            conn = get_db()

            # Tier distribution
            tier_dist = conn.execute("""
                SELECT dispatch_priority, COUNT(*) as count
                FROM memories
                WHERE active = 1
                GROUP BY dispatch_priority
                ORDER BY dispatch_priority
            """).fetchall()

            print("Tier Distribution:")
            for row in tier_dist:
                tier = row['dispatch_priority']
                count = row['count']
                print(f"  Tier {tier}: {count} memories")

            # Locked memories (cooldown)
            locked = conn.execute("""
                SELECT COUNT(*) as count
                FROM memories
                WHERE active = 1
                  AND tier_locked_until IS NOT NULL
                  AND tier_locked_until > datetime('now')
            """).fetchone()['count']
            print(f"\nLocked (cooldown): {locked} memories")

            # Pending promotions
            pending2 = conn.execute("""
                SELECT COUNT(*) as count
                FROM memories
                WHERE active = 1
                  AND promotion_signals = 2
            """).fetchone()['count']

            pending1 = conn.execute("""
                SELECT COUNT(*) as count
                FROM memories
                WHERE active = 1
                  AND promotion_signals = 1
            """).fetchone()['count']

            print(f"\nPending promotion (2 signals): {pending2} memories")
            print(f"Pending promotion (1 signal): {pending1} memories")

            # Tier 0 at-risk (>60 days no access)
            cutoff_60d = (datetime.now() - timedelta(days=60)).isoformat()
            at_risk = conn.execute("""
                SELECT COUNT(*) as count
                FROM memories
                WHERE active = 1
                  AND dispatch_priority = 0
                  AND (last_accessed_at IS NULL OR last_accessed_at < ?)
            """, (cutoff_60d,)).fetchone()['count']
            print(f"\nTier 0 at-risk (>60 days no access): {at_risk} memories")

            conn.close()

        else:
            print(f"Unknown subcommand: {subcmd}")
            print("\nAvailable subcommands:")
            print("  run    - Run tier evolution batch job")
            print("  stats  - Show tier distribution and evolution stats")

    elif cmd == "contradict" and len(sys.argv) >= 3:
        from .contradiction import log_contradiction
        flags, args = parse_flags(sys.argv, 2)

        if not args:
            print("Usage: memory-tool contradict <memory_id> [--expected TEXT] [--found TEXT] [--context-hash HASH]")
            sys.exit(1)

        memory_id = int(args[0])
        expected = flags.get("expected")
        found = flags.get("found")
        context_hash = flags.get("context-hash")

        contradiction_id = log_contradiction(
            memory_id=memory_id,
            expected=expected,
            found=found,
            context_hash=context_hash
        )
        print(f"Logged contradiction #{contradiction_id} for memory #{memory_id}")

    elif cmd == "reconcile":
        from .contradiction import reconcile
        flags, _ = parse_flags(sys.argv, 2)
        dry_run = flags.get("dry-run", False)

        if dry_run:
            print("DRY RUN MODE: No changes will be made\n")

        result = reconcile(dry_run=dry_run)

        print("\nReconciliation complete:")
        if result['deleted']:
            print(f"  Deleted: {len(result['deleted'])} memories")
            for mem_id in result['deleted']:
                print(f"    #{mem_id}")
        if result['flagged']:
            print(f"  Flagged for review: {len(result['flagged'])} memories")
            for mem_id in result['flagged']:
                print(f"    #{mem_id}")
        if result['skipped']:
            print(f"  Skipped: {result['skipped']} (already inactive or deleted)")

        if not result['deleted'] and not result['flagged'] and not result['skipped']:
            print("  No contradictions to reconcile.")

    elif cmd == "contradictions":
        from .contradiction import show_contradiction_journal
        flags, _ = parse_flags(sys.argv, 2)
        limit = int(flags.get("limit", 20))
        show_contradiction_journal(limit=limit)

    elif cmd == "believe" and len(sys.argv) >= 3:
        from .beliefs import set_confidence
        statement = " ".join(sys.argv[2:])
        # Parse flags if present
        flags, content_parts = parse_flags(sys.argv, 2)
        if content_parts:
            statement = " ".join(content_parts)

        confidence = float(flags.get("confidence", 0.7))
        based_on = int(flags.get("based-on")) if flags.get("based-on") else None

        # Add as belief memory
        mem_id = add_memory(
            "belief",
            statement,
            tags="belief,explicit",
            project=flags.get("project")
        )

        # Set explicit confidence
        conn = get_db()
        set_confidence(conn, mem_id, confidence, "Explicit belief creation")
        conn.close()

        print(f"Created belief #{mem_id} with confidence {confidence:.2f}")

    elif cmd == "predict" and len(sys.argv) >= 3:
        from .beliefs import predict
        flags, content_parts = parse_flags(sys.argv, 2)
        prediction = " ".join(content_parts) if content_parts else " ".join(sys.argv[2:])

        based_on = int(flags.get("based-on")) if flags.get("based-on") else None
        confidence = float(flags.get("confidence", 0.5))
        deadline = flags.get("deadline")
        expected = flags.get("expect", "")

        conn = get_db()
        pred_id = predict(conn, prediction, based_on, confidence, deadline, expected)
        conn.close()

        print(f"Created prediction #{pred_id}")
        if deadline:
            print(f"  Deadline: {deadline}")
        print(f"  Confidence: {confidence:.2f}")

    elif cmd == "resolve" and len(sys.argv) >= 3:
        from .beliefs import resolve_prediction
        pred_id = int(sys.argv[2])
        flags, outcome_parts = parse_flags(sys.argv, 3)

        confirmed = flags.get("confirmed", False)
        refuted = flags.get("refuted", False)

        if not confirmed and not refuted:
            print("Error: Must specify --confirmed or --refuted")
            sys.exit(1)

        outcome = " ".join(outcome_parts) if outcome_parts else flags.get("outcome", "")

        conn = get_db()
        result = resolve_prediction(conn, pred_id, outcome, confirmed)
        conn.close()

        status = "confirmed" if confirmed else "refuted"
        print(f"Resolved prediction #{pred_id} as {status}")
        print(f"  Updated {result['updated']} memories via belief propagation")

    elif cmd == "beliefs":
        from .beliefs import weakest_beliefs, strongest_beliefs, belief_conflicts
        conn = get_db()

        # Check for --state filter
        if "--state" in sys.argv:
            state_idx = sys.argv.index("--state")
            if state_idx + 1 < len(sys.argv):
                state = sys.argv[state_idx + 1]
                try:
                    from .beliefs import list_beliefs_by_state
                    rows = list_beliefs_by_state(conn, state)
                    print(f"Beliefs (state: {state})")
                    print("=" * 70)
                    for r in rows:
                        stmt = r['statement'][:60] + "..." if len(r['statement']) > 60 else r['statement']
                        print(f"  #{r['id']:>3} {r['confidence']:.2f} [{r['category']:<8}] {stmt}")
                    print(f"\n({len(rows)} beliefs in state '{state}')")
                except ImportError:
                    print("Extended beliefs system not available")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Error: --state requires a value (hypothesis/tested/validated/deprecated/refuted)")
            conn.close()

        elif "--weak" in sys.argv:
            rows = weakest_beliefs(conn, 20)
            print("Weakest Beliefs (lowest confidence)")
            print("=" * 70)
            for r in rows:
                content = r['content'][:60] + "..." if len(r['content']) > 60 else r['content']
                print(f"  #{r['id']:>3} {r['confidence']:.2f} [{r['category']:<8}] {content}")
            print(f"\n({len(rows)} beliefs)")
            conn.close()

        elif "--strong" in sys.argv:
            rows = strongest_beliefs(conn, 20)
            print("Strongest Beliefs (highest confidence)")
            print("=" * 70)
            for r in rows:
                content = r['content'][:60] + "..." if len(r['content']) > 60 else r['content']
                print(f"  #{r['id']:>3} {r['confidence']:.2f} [{r['category']:<8}] {content}")
            print(f"\n({len(rows)} beliefs)")
            conn.close()

        elif "--conflicts" in sys.argv:
            conflicts = belief_conflicts(conn)
            if conflicts:
                print("Belief Conflicts (high-confidence contradictions)")
                print("=" * 70)
                for c in conflicts:
                    print(f"  #{c['id1']} vs #{c['id2']} ({c['similarity']:.0%} similar)")
                    print(f"    A ({c['confidence1']:.2f}): {c['content1'][:60]}...")
                    print(f"    B ({c['confidence2']:.2f}): {c['content2'][:60]}...")
                    print()
                print(f"({len(conflicts)} conflicts)")
            else:
                print("No belief conflicts found.")
            conn.close()

        else:
            # Show all beliefs sorted by confidence
            rows = conn.execute("""
                SELECT id, category, content, confidence, access_count
                FROM memories
                WHERE active = 1 AND category IN ('belief', 'learning', 'decision')
                ORDER BY confidence DESC
                LIMIT 30
            """).fetchall()

            print("Beliefs (sorted by confidence)")
            print("=" * 70)
            for r in rows:
                content = r['content'][:60] + "..." if len(r['content']) > 60 else r['content']
                bar = "█" * int(r['confidence'] * 10) + "░" * (10 - int(r['confidence'] * 10))
                print(f"  #{r['id']:>3} {bar} {r['confidence']:.2f} [{r['category']:<8}] {content}")
            print(f"\n({len(rows)} beliefs)")
            conn.close()

    elif cmd == "lifecycle" and len(sys.argv) >= 4:
        # memory-tool lifecycle <id> <state>
        from .beliefs import set_belief_state
        belief_id = int(sys.argv[2])
        new_state = sys.argv[3]
        reason = " ".join(sys.argv[4:]) if len(sys.argv) > 4 else None

        conn = get_db()
        try:
            set_belief_state(conn, belief_id, new_state, reason)
            print(f"✓ Belief #{belief_id} transitioned to '{new_state}'")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        conn.close()

    elif cmd == "timeline":
        # memory-tool timeline [--project X] [--days N] [<id>]
        from .beliefs import get_timeline, get_confidence_history, format_timeline_entry, get_timeline_summary
        flags, args = parse_flags(sys.argv, 2)

        conn = get_db()

        # Check if specific ID provided
        if args:
            identifier = int(args[0])
            # Try to determine if it's a belief or memory ID
            is_belief = conn.execute("SELECT 1 FROM beliefs WHERE id = ?", (identifier,)).fetchone() is not None

            history = get_confidence_history(conn, identifier, is_belief)

            entity_type = "Belief" if is_belief else "Memory"
            print(f"{entity_type} #{identifier} Confidence History")
            print("=" * 70)

            if history:
                for entry in history:
                    print(f"  {format_timeline_entry(entry)}")
                print(f"\n({len(history)} changes)")
            else:
                print(f"No confidence changes recorded for {entity_type.lower()} #{identifier}")
        else:
            # Show general timeline
            project = flags.get("project")
            days = int(flags.get("days", 30))

            entries = get_timeline(conn, project=project, days=days)

            print(f"Timeline (last {days} days{f', project: {project}' if project else ''})")
            print("=" * 70)

            if entries:
                for entry in entries:
                    # Format entry with context
                    if entry['belief_statement']:
                        context = entry['belief_statement'][:40] + "..."
                    elif entry['memory_content']:
                        context = entry['memory_content'][:40] + "..."
                    else:
                        context = "Unknown"

                    print(f"  {format_timeline_entry(entry)}")
                    print(f"    Context: {context}")

                print(f"\n({len(entries)} changes)")

                # Show summary stats
                summary = get_timeline_summary(conn, days)
                print(f"\nSummary:")
                print(f"  Total changes: {summary['total_changes']}")
                print(f"  Increases: {summary['increases']} ↑")
                print(f"  Decreases: {summary['decreases']} ↓")
                print(f"  Avg delta: {summary['avg_confidence_delta']:+.3f}")
                if summary['by_source']:
                    by_source_str = ', '.join([f"{s['source_type']}({s['count']})" for s in summary['by_source']])
                print(f"  By source: {by_source_str}")
            else:
                print(f"No timeline entries in the last {days} days.")

        conn.close()

    elif cmd == "predictions":
        from .beliefs import list_predictions, expired_predictions
        conn = get_db()

        if "--open" in sys.argv:
            preds = list_predictions(conn, 'open')
            status_filter = 'open'
        elif "--confirmed" in sys.argv:
            preds = list_predictions(conn, 'confirmed')
            status_filter = 'confirmed'
        elif "--refuted" in sys.argv:
            preds = list_predictions(conn, 'refuted')
            status_filter = 'refuted'
        elif "--expired" in sys.argv:
            preds = expired_predictions(conn)
            status_filter = 'expired'
        else:
            preds = list_predictions(conn, 'all')
            status_filter = 'all'

        if preds:
            print(f"Predictions ({status_filter})")
            print("=" * 70)
            for p in preds:
                deadline_str = f" [deadline: {p['deadline']}]" if p['deadline'] else ""
                mem_ref = f" (based on #{p['memory_id']})" if p['memory_id'] else ""
                pred_preview = p['prediction'][:50] + "..." if len(p['prediction']) > 50 else p['prediction']
                print(f"  #{p['id']:>3} [{p['status']:<10}] {p['confidence']:.2f} {pred_preview}{deadline_str}{mem_ref}")
            print(f"\n({len(preds)} predictions)")
        else:
            print(f"No {status_filter} predictions.")

        conn.close()

    # Extended beliefs system commands
    elif cmd == "believe" and len(sys.argv) >= 3:
        from .beliefs import add_belief
        flags, statement_parts = parse_flags(sys.argv, 2)
        statement = " ".join(statement_parts)

        conn = get_db()
        belief_id = add_belief(
            conn,
            statement,
            confidence=float(flags.get("confidence", 0.5)),
            category=flags.get("category", "general"),
            source=flags.get("source", "user"),
            memory_id=int(flags["memory"]) if "memory" in flags else None
        )
        conn.close()
        print(f"Created belief #{belief_id}")

    elif cmd == "evidence" and len(sys.argv) >= 4:
        from .beliefs import add_evidence
        belief_id = int(sys.argv[2])
        memory_id = int(sys.argv[3])
        flags, _ = parse_flags(sys.argv, 4)

        direction = "supports" if "supports" in sys.argv else "contradicts"
        strength = float(flags.get("strength", 0.5))
        note = flags.get("note")

        conn = get_db()
        add_evidence(conn, belief_id, memory_id, direction, strength, note)
        conn.close()
        print(f"Added {direction} evidence to belief #{belief_id}")

    elif cmd == "belief-stats":
        from .beliefs import belief_accuracy, strongest_beliefs_extended, weakest_beliefs_extended, most_revised
        conn = get_db()

        print("=== Belief System Statistics ===\n")

        # Prediction accuracy
        acc = belief_accuracy(conn)
        if acc['total_predictions'] > 0:
            print(f"Prediction Accuracy:")
            print(f"  Total: {acc['total_predictions']} ({acc['correct_count']} correct, {acc['incorrect_count']} incorrect)")
            print(f"  Accuracy: {acc['correct_percentage']:.1f}%")
            print(f"  Avg confidence (correct): {acc['avg_confidence_correct']:.2f}")
            print(f"  Avg confidence (incorrect): {acc['avg_confidence_incorrect']:.2f}")

            if acc['calibration']:
                print(f"\n  Calibration by confidence bucket:")
                for bucket, stats in acc['calibration'].items():
                    print(f"    {bucket}: {stats['accuracy']:.1%} actual vs {stats['expected']:.1%} expected ({stats['count']} predictions)")

        # Strongest beliefs
        print(f"\nStrongest Beliefs:")
        strong = strongest_beliefs_extended(conn, 5)
        for b in strong:
            print(f"  #{b['id']} {b['confidence']:.2f} [{b['category']}] {b['statement'][:60]}")

        # Weakest beliefs
        print(f"\nWeakest Beliefs:")
        weak = weakest_beliefs_extended(conn, 5)
        for b in weak:
            print(f"  #{b['id']} {b['confidence']:.2f} [{b['category']}] {b['statement'][:60]}")

        # Most revised (controversial)
        print(f"\nMost Revised (Controversial):")
        revised = most_revised(conn, 5)
        for b in revised:
            print(f"  #{b['id']} {b['revision_count']} revisions {b['confidence']:.2f} [{b['category']}] {b['statement'][:60]}")

        conn.close()

    elif cmd == "expired-predictions":
        from .beliefs import check_expired_predictions
        conn = get_db()
        expired = check_expired_predictions(conn)
        conn.close()

        if expired:
            print(f"Found {len(expired)} expired predictions:\n")
            for p in expired:
                print(f"  #{p['id']} deadline: {p['deadline']}")
                print(f"    {p['prediction'][:80]}")
                print()
        else:
            print("No expired predictions.")

    elif cmd == "narrative" and len(sys.argv) >= 3:
        from .narrative import build_narrative, get_entity_stories, get_causal_chains
        entity_name = " ".join(sys.argv[2:])
        flags, name_parts = parse_flags(sys.argv, 2)
        if name_parts:
            entity_name = " ".join(name_parts)

        conn = get_db()
        result = build_narrative(conn, entity_name, max_depth=int(flags.get("depth", 3)))

        if result['entity']:
            print(result['narrative'])

            # Show causal chains
            chains = get_causal_chains(conn, entity_name)
            if chains:
                print(f"\nCausal chains ({len(chains)}):")
                for chain in chains[:5]:
                    print(f"  {' → '.join(chain)}")
        else:
            print(f"Entity '{entity_name}' not found.")
            # Show entities with richest stories
            stories = get_entity_stories(conn, 5)
            if stories:
                print("\nEntities with narratives:")
                for s in stories:
                    print(f"  {s['name']} ({s['type']}) — {s['rel_count']} relationships, {s['mem_count']} memories")
        conn.close()

    elif cmd == "meta":
        from .meta_learning import get_meta_stats, apply_learned_weights
        conn = get_db()

        if "--apply" in sys.argv:
            result = apply_learned_weights(conn)
            if result['applied']:
                print("✓ Search weights updated:")
                print(f"  Keyword: {result['old_weights']['keyword_weight']:.2f} → {result['new_weights']['keyword_weight']:.2f}")
                print(f"  Semantic: {result['old_weights']['semantic_weight']:.2f} → {result['new_weights']['semantic_weight']:.2f}")
                print(f"  Reason: {result['recommendation']}")
            else:
                print(f"No changes applied: {result['reason']}")
        else:
            stats = get_meta_stats(conn)
            print("Meta-Learning Report")
            print("=" * 70)
            print(f"\nCurrent weights:")
            for k, v in stats['current_weights'].items():
                print(f"  {k}: {v:.2f}")

            eff = stats['effectiveness_30d']
            print(f"\nLast 30 days ({eff['total_searches']} searches):")
            print(f"  Keyword effectiveness: {eff['keyword_effectiveness']:.1%}")
            print(f"  Semantic effectiveness: {eff['semantic_effectiveness']:.1%}")
            print(f"  Overall hit rate: {eff['overall_hit_rate']:.1%}")
            print(f"  Recommendation: {eff['recommendation']}")

            if eff['recommendation'] not in ('balanced', 'insufficient_data'):
                print(f"\nSuggested weights:")
                for k, v in eff['suggested_weights'].items():
                    curr = stats['current_weights'].get(k, 0)
                    delta = v - curr
                    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                    print(f"  {k}: {v:.2f} ({arrow}{abs(delta):.2f})")
                print(f"\nRun 'memory-tool meta --apply' to apply suggested weights")

            print(f"\nWeight changes: {stats['weight_changes']} total")
        conn.close()

    elif cmd == "identity":
        from .identity import discover_traits, get_identity, save_identity_snapshot, compare_identity_snapshots
        conn = get_db()

        if "--discover" in sys.argv or "--refresh" in sys.argv:
            print("Discovering traits from memories...\n")
            traits = discover_traits(conn)
            save_identity_snapshot(conn)

            if traits:
                print(f"Found {len(traits)} traits:\n")
                for t in traits:
                    bar = "█" * int(t['confidence'] * 10) + "░" * (10 - int(t['confidence'] * 10))
                    print(f"  {bar} {t['confidence']:.0%}  {t['description']}")
                    print(f"         Evidence: {t['evidence_count']} for, {t['counter_evidence']} against")
            else:
                print("No traits discovered. Add more memories first.")

        elif "--evolution" in sys.argv:
            result = compare_identity_snapshots(conn)
            if result['has_comparison']:
                print(f"Identity Evolution: {result['previous_date'][:10]} → {result['current_date'][:10]}")
                print("=" * 70)

                if result['new_traits']:
                    print(f"\nNew traits ({len(result['new_traits'])}):")
                    for t in result['new_traits']:
                        print(f"  + {t['trait_name']} ({t['confidence']:.0%})")

                if result['removed_traits']:
                    print(f"\nRemoved traits ({len(result['removed_traits'])}):")
                    for t in result['removed_traits']:
                        print(f"  - {t['trait_name']}")

                if result['changed']:
                    print(f"\nConfidence changes:")
                    for c in result['changed']:
                        arrow = "↑" if c['delta'] > 0 else "↓"
                        print(f"  {arrow} {c['trait_name']}: {c['old_confidence']:.0%} → {c['new_confidence']:.0%}")

                if not result['new_traits'] and not result['removed_traits'] and not result['changed']:
                    print("\nNo significant changes between snapshots.")
            else:
                print(result['message'])
                print("Run 'memory-tool identity --discover' to create first snapshot.")

        else:
            # Show current identity
            identity = get_identity(conn)

            if identity['traits']:
                print("Identity Profile")
                print("=" * 70)
                print(f"\n{identity['summary']}")
                print(f"\nTotal: {identity['total_traits']} traits, {identity['total_evidence']} evidence points")
                print(f"  Strong: {len(identity['strong'])} | Moderate: {len(identity['moderate'])} | Weak: {len(identity['weak'])}")
            else:
                print("No identity profile yet.")
                print("Run 'memory-tool identity --discover' to build profile from memories.")

        conn.close()

    elif cmd == "user-model":
        # Honcho-style user modeling
        cmd_user_model(sys.argv[1:])

    elif cmd == "session-log":
        from pathlib import Path

        SESSION_LOG = Path("/tmp/ai-iq-session-log.jsonl")

        if not SESSION_LOG.exists():
            print("No session log found. Hook not yet active or no tools executed.")
            print(f"To enable, install hook: /root/ai-iq/hooks/session-logger.mjs")
            sys.exit(0)

        # Read and display session log
        flags, _ = parse_flags(sys.argv, 2)
        limit = int(flags.get("limit", 50))  # Default last 50 entries
        show_errors_only = flags.get("errors", False)

        entries = []
        with open(SESSION_LOG, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        # Filter if errors-only mode
        if show_errors_only:
            entries = [e for e in entries if e.get('tool') == 'Bash' and e.get('exit_code') != 0]

        # Show last N entries
        recent = entries[-limit:]

        print(f"Session log ({len(recent)} of {len(entries)} total entries):")
        print()

        for e in recent:
            ts = e.get('timestamp', '')[:19]  # Trim milliseconds
            tool = e.get('tool', 'Unknown')

            if tool == 'Bash':
                cmd = e.get('input', '')
                exit_code = e.get('exit_code', '?')
                status = "✓" if exit_code == 0 else "✗"
                print(f"{ts} [{tool}] {status} {cmd}")
                if exit_code != 0:
                    preview = e.get('output_preview', '')
                    if preview:
                        print(f"    Error: {preview}")
            elif tool in ('Read', 'Edit', 'Write'):
                file_path = e.get('file_path', '')
                action = e.get('action', '')
                print(f"{ts} [{tool}] {action} {file_path}")
            elif tool in ('Glob', 'Grep'):
                pattern = e.get('pattern', '')
                preview = e.get('output_preview', '')
                print(f"{ts} [{tool}] {pattern} → {preview}")
            else:
                inp = e.get('input', '')[:50]
                preview = e.get('output_preview', '')[:50]
                print(f"{ts} [{tool}] {inp} → {preview}")

        print()
        print(f"Use --limit N to show more, --errors to show only failures")

    elif cmd == "mode":
        from .modes import get_mode, set_mode, list_modes

        if len(sys.argv) == 2:
            # Show current mode
            current = get_mode()
            print(f"Current mode: {current}")
            print("\nRun 'memory-tool mode list' to see all modes")
            print("Run 'memory-tool mode <name>' to switch")

        elif sys.argv[2] == "list":
            # List all modes
            list_modes()

        else:
            # Set mode
            mode_name = sys.argv[2]
            if set_mode(mode_name):
                print(f"✅ Switched to mode: {mode_name}")
            else:
                print(f"❌ Failed to set mode. Valid modes: default, dev, ops, research, monitor")
                sys.exit(1)

    elif cmd == "passport":
        from .passport_w3c import cmd_passport
        conn = get_db()
        cmd_passport(sys.argv[2:], conn)
        conn.close()

    elif cmd == "verify-passport" and len(sys.argv) >= 3:
        from .passport_w3c import cmd_verify_passport
        conn = get_db()
        cmd_verify_passport(sys.argv[2:], conn)
        conn.close()

    elif cmd == "patterns":
        from .patterns import add_pattern, list_patterns, brief, get_stats
        subcmd = sys.argv[2] if len(sys.argv) > 2 else "list"
        if subcmd == "add":
            adopt = None
            avoid = None
            context = ""
            args = sys.argv[3:]
            i = 0
            while i < len(args):
                if args[i] == "--adopt" and i + 1 < len(args):
                    adopt = args[i + 1]; i += 2
                elif args[i] == "--avoid" and i + 1 < len(args):
                    avoid = args[i + 1]; i += 2
                elif args[i] == "--context" and i + 1 < len(args):
                    context = args[i + 1]; i += 2
                else:
                    i += 1
            if not adopt and not avoid:
                print("Usage: memory-tool patterns add --adopt \"X\" [--avoid \"Y\"] [--context \"Z\"]")
                sys.exit(1)
            rec = add_pattern(adopt=adopt, avoid=avoid, context=context)
            print(f"✅ Pattern recorded [{rec['session_id']}]")
            if adopt: print(f"  ADOPT: {adopt}")
            if avoid: print(f"  AVOID: {avoid}")
        elif subcmd == "brief":
            context = sys.argv[3] if len(sys.argv) > 3 else ""
            print(brief(context=context))
        elif subcmd == "stats":
            s = get_stats()
            print(f"📊 Pattern Store Stats")
            print(f"  Sessions tracked : {s['total_sessions']}")
            print(f"  Adopt patterns   : {s['adopt_count']}")
            print(f"  Avoid patterns   : {s['avoid_count']}")
            print(f"  Top contexts     : {', '.join(s['top_context_keywords']) or 'none'}")
            print(f"  Last recorded    : {s['last_recorded'] or 'never'}")
            print(f"  File             : {s['patterns_file']}")
        else:  # list
            last_n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
            rows = list_patterns(last_n=last_n)
            if not rows:
                print("No patterns recorded yet.")
                print("  memory-tool patterns add --adopt \"what worked\" --avoid \"what didn't\"")
            else:
                print(f"📚 Last {len(rows)} session patterns (newest first):")
                for p in rows:
                    date = p.get("recorded_at", "")[:10]
                    ctx = f" [{p['context']}]" if p.get("context") else ""
                    for item in p.get("adopt", []):
                        print(f"  ✅ {date}{ctx}: {item}")
                    for item in p.get("avoid", []):
                        print(f"  ❌ {date}{ctx}: {item}")

    elif cmd == "ssl-finetune":
        from .ssl_finetune import finetune_on_memories
        flags, _ = parse_flags(sys.argv, 2)
        epochs = int(flags.get("epochs", 3))
        batch_size = int(flags.get("batch-size", 32))
        dry_run = flags.get("dry-run", False)

        print("Starting SimCSE fine-tuning on memory corpus...")
        if dry_run:
            print("[dry-run] No training will occur")
        ok = finetune_on_memories(epochs=epochs, batch_size=batch_size, dry_run=dry_run)
        if ok:
            print("Done. Run 'memory-tool reembed' to re-index all memories.")
        else:
            print("Fine-tuning skipped (too few memories or error).", file=sys.stderr)
            sys.exit(1)

    elif cmd == "reembed":
        from .embedding import reindex_embeddings
        flags, _ = parse_flags(sys.argv, 2)
        confirm = flags.get("confirm", False)

        if not confirm:
            print("This will re-embed all active memories with the current model.")
            response = input("Continue? [y/N]: ")
            if response.lower() not in ('y', 'yes'):
                print("Aborted.")
                sys.exit(0)

        conn = get_db()
        print("Re-embedding all memories...")
        reindex_embeddings(conn)
        conn.close()
        print("Done.")

    elif cmd == "serve":
        # HTTP API server mode
        flags, _ = parse_flags(sys.argv, 2)
        port = int(flags.get("port", 37777))
        host = flags.get("host", "127.0.0.1")

        from .server import run_server
        run_server(host, port)

    elif cmd in ("help", "--help", "-h"):
        print_help()

    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)

