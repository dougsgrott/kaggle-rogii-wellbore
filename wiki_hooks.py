"""Competition-specific hooks for wikikit (enable via [hooks] in competition.toml).

Define any of the hook functions below; whatever this file doesn't define
falls back to the package default (wikikit.hooks). Keep the logic in sync with
the [[archetype]] tables in competition.toml — Profile.hard(name) raises a
helpful error when they drift apart.
"""
from wikikit.profile import Profile


# def score_archetypes(feat: dict, profile: Profile) -> dict[str, float]:
#     """Weighted archetype scores from a Phase-2 feature record.
#
#     `feat` fields you'll typically use:
#       keyword_hits  — {keyword: count} over the lexicon + hard sets
#       import_tags   — categories from the import map ("learned", "numeric", ...)
#       lexicon       — {category: total hits}
#       parsed / n_funcs / cc_max — structure signals
#
#     The default scorer is 2.0 * distinct hard-keyword hits per archetype;
#     override when you need import corroboration, floors, or exclusions, e.g.:
#     """
#     kw = feat.get("keyword_hits", {})
#     s = {a.name: 2.0 * sum(1 for w in a.hard if w in kw)
#          for a in profile.archetypes}
#     # example: structured code with no dominant strategy gets a floor
#     # if feat.get("parsed") and feat.get("n_funcs", 0) >= 3:
#     #     s["rule-based/heuristic"] = 0.75
#     return s
