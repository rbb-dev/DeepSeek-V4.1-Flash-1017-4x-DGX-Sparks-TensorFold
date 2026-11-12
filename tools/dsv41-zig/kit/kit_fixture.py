"""kit_fixture.py KIT_DIR OUT_DIR: the kit's variants of the drafter stage's attention launches and the twins, for
draft.zig's test that a four-way stage's launches resolve in the kit, and tri_attn.zig's and tri_index.zig's that every
_sparse_attn_part and _index_score_rows variant has its twin.
For each of _rope_heads, _kv_norm_rope, _sparse_attn_part, _sparse_attn_merge, _sparse_attn_part_u, _index_score_rows,
_index_score_rows_u and _index_score_rows_v: every entry of
KIT_DIR/aot/aot.json into OUT_DIR/kit-drafter.json and every entry of KIT_DIR/aot-extra/aot.json (none when the kit has
no extra set) into OUT_DIR/kit-drafter-extra.json, in the set's order, with the fields aot.zig matches (fn, hash, name,
num_warps, params, consts): each file an aot.json of its own. Standard library only."""
FUNCTIONS = ("_rope_heads", "_kv_norm_rope", "_sparse_attn_part", "_sparse_attn_merge", "_sparse_attn_part_u",
             "_index_score_rows", "_index_score_rows_u", "_index_score_rows_v")