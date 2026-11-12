#!/usr/bin/env python3
"""The schema generator's build checks: a row an agent may set must be bounded, every row sits on a listed tab, a cost says
measured or estimated, a requires names a row. Run from the engine's root: python3 tools/schema/test_gen_schema.py"""
import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import gen_schema  # noqa: E402

D = json.load(open(gen_schema.SCHEMA))


def row(name):
    return next(r for r in D["options"] if r["name"] == name)


class Checks(unittest.TestCase):
    def mutate(self, name, **kw):
        d = copy.deepcopy(D)
        next(r for r in d["options"] if r["name"] == name).update(kw)
        return d

    def test_the_shipped_schema_renders(self):
        gen_schema.render(D)

    def test_an_agent_row_without_bounds_is_refused(self):
        r = row("tf_ds_rounds_beside")
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate(r["name"], who=["admin", "agent"]))

    def test_an_agent_row_with_bounds_renders_into_the_table(self):
        r = row("tf_ds_rounds_beside")
        d = self.mutate(r["name"], who=["admin", "agent"], min=1, max=8, agent={"min": 1, "max": 4, "max_ttl_s": 3600})
        out = gen_schema.render(d)
        self.assertIn(".agent = .{ .min = 1.0, .max = 4.0, .max_ttl_s = 3600 }", out)

    def test_bounds_without_who_agent_are_refused(self):
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate("tf_ds_rounds_beside", agent={"min": 1, "max": 4, "max_ttl_s": 60}))

    def test_a_row_on_an_unlisted_tab_is_refused(self):
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate("tf_ds_rounds_beside", tab="Nowhere"))

    def test_a_cost_is_measured_or_estimated(self):
        c = {"what": "x", "value": "1", "label": "guessed", "source": "s"}
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate("tf_ds_rounds_beside", cost=[c]))
        c["label"] = "estimated"
        self.assertIn('.label = "estimated"', gen_schema.render(self.mutate("tf_ds_rounds_beside", cost=[c])))

    def test_requires_names_a_row(self):
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate("tf_ds_rounds_beside", requires=["no_such_row=1"]))
        gen_schema.render(self.mutate("tf_ds_rounds_beside", requires=["tf_ds_window=1048576"]))

    def test_every_row_has_a_label_and_a_doc_that_names_no_variable(self):
        for r in D["options"]:
            self.assertTrue(r["label"] and len(r["label"]) <= 40 and "_" not in r["label"], r["name"])
            self.assertTrue(r["doc"].strip() and r["env"] not in r["doc"], r["name"])

    def test_a_row_without_a_label_or_with_an_env_name_in_its_doc_is_refused(self):
        r = row("tf_ds_rounds_beside")
        for kw in (dict(label=""), dict(label="Has_underscore"), dict(label="x" * 41), dict(doc=""), dict(doc="set %s to 1" % r["env"])):
            with self.assertRaises(AssertionError, msg=str(kw)):
                gen_schema.render(self.mutate(r["name"], **kw))

    def test_every_row_has_a_default_or_the_words_for_one(self):
        for r in D["options"]:
            self.assertTrue(r["default"] != "" or r.get("default_text"), r["name"])

    def test_rows_out_of_id_order_are_refused(self):
        # settings.byId is a binary search over the table in schema order: a row out of id order is never found
        d = copy.deepcopy(D)
        d["options"][-1], d["options"][-2] = d["options"][-2], d["options"][-1]
        with self.assertRaises(AssertionError):
            gen_schema.render(d)

    def test_an_empty_default_without_words_is_refused(self):
        with self.assertRaises(AssertionError):
            gen_schema.render(self.mutate("tf_ds_window", default="", default_text=""))
        self.assertIn('.default_text = "Same as the context window"', gen_schema.render(D))

    def test_a_choice_lists_its_words_and_its_default_is_one_of_them(self):
        for r in D["options"]:
            if r["kind"] == "enum":
                self.assertTrue(r["values"], r["name"])
                self.assertTrue(r["default"] == "" or r["default"] in r["values"], r["name"])
            else:
                self.assertEqual(r["values"], [], r["name"])

    def test_the_scan_artifacts_are_not_rows(self):
        envs = {r["env"] for r in D["options"]}
        self.assertFalse(envs & {"TF_DS_", "TF_DS_SPILL_", "TF_SPILL_"})


if __name__ == "__main__":
    unittest.main()
