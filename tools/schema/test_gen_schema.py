    def test_rows_out_of_id_order_are_refused(self):
        # settings.byId is a binary search over the table in schema order: a row out of id order is never found
        d = copy.deepcopy(D)
        d["options"][-1], d["options"][-2] = d["options"][-2], d["options"][-1]
        with self.assertRaises(AssertionError):
            gen_schema.render(d)
