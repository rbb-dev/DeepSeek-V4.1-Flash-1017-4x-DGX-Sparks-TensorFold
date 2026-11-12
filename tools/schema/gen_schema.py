    ids = [r["id"] for r in rows]
    assert all(a < b for a, b in zip(ids, ids[1:])), "rows must be in rising id order (settings.byId is a binary search over the table): %s" % next("%d before %d" % (a, b) for a, b in zip(ids, ids[1:]) if a >= b)
    out.append("};\n\n")
    out.append("/// Names that no longer exist: a start with one set (the environment or engine.cfg) is refused by name.\n")
    out.append("pub const retired = [_]settings.Retired{\n")
    for r in d.get("retired", []):
        out.append("    .{ .env = %s, .when = .%s, .why = %s },\n" % (z(r["env"]), r["when"], z(r["why"])))