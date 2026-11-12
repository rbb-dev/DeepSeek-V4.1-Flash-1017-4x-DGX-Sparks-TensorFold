    out.append("};\n\n")
    out.append("/// Names that no longer exist: a start with one set (the environment or engine.cfg) is refused by name.\n")
    out.append("pub const retired = [_]settings.Retired{\n")
    for r in d.get("retired", []):
        out.append("    .{ .env = %s, .when = .%s, .why = %s },\n" % (z(r["env"]), r["when"], z(r["why"])))