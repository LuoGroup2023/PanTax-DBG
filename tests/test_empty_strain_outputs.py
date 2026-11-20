"""Regression tests for valid no-strain outcomes; no native tools are needed."""

import csv
import tempfile
import unittest
from pathlib import Path

from pantax_dbg_scripts import add_strain_taxid_and_name_columns as annotate
from pantax_dbg_scripts import filter_and_topk_strains_per_species as topk
from pantax_dbg_scripts import strain_postprocess_global_abundance as post
from pantax_dbg_scripts import tax_profile_strain_update as tree


BASE = ["species", "group_id", "assigned_reads", "rel_abundance_within_species", "strains"]
FULL = BASE + ["strain_taxid", "strain_name"]


class EmptyStrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mix = self.root / "species.tsv"
        self.mix.write_text("speciesID\tabundance\n10\t0.75\n20\t0.25\n")

    def table(self, name, fields=FULL, rows=()):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(fields)
            writer.writerows(rows)
        return str(path)

    def read(self, path):
        with open(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return reader.fieldnames, list(reader)

    def refine(self, input, **kwargs):
        args = dict(mix=str(self.mix), input=input, threshold=10,
                    out_prefix=str(self.root / "final"), div=5, emit_name=True)
        args.update(kwargs)
        return post.run(**args)

    def test_topk_empty_preserves_identity_columns_and_extra_fields(self):
        input = self.table("input.tsv", FULL + ["extra"])
        out = str(self.root / "topk.tsv")
        topk.run(input=input, out=out, min_reads=5, topk=3)
        self.assertEqual(self.read(out), (FULL + ["extra"], []))

    def test_exact_minimum_is_still_excluded(self):
        input = self.table("input.tsv", rows=[
            [10, "G1", 5, 0.5, "10_1", 101, "s1"],
            [10, "G2", 5.01, 0.5, "10_2", 102, "s2"],
        ])
        out = str(self.root / "topk.tsv")
        topk.run(input=input, out=out, min_reads=5, topk=3)
        self.assertEqual([r["strain_taxid"] for r in self.read(out)[1]], ["102"])

    def test_empty_chain_preserves_species_file_and_writes_empty_tree(self):
        mapping = self.table("map.tsv", ["new_strain_taxid", "strain_taxid", "strain_name"],
                             [["10_1", 101, "s1"]])
        input = self.table("raw.tsv", BASE, [[10, "G1", 4, 1, "10_1"]])
        with_ids = str(self.root / "ids.tsv")
        selected = str(self.root / "topk.tsv")
        annotate.run(input=input, map=mapping, out=with_ids)
        topk.run(input=with_ids, out=selected, min_reads=5, topk=3)
        species_before = self.mix.read_bytes()
        detailed, abundance = self.refine(selected)
        self.assertEqual(self.read(abundance), (["strain_taxid", "strain_name", "abundance"], []))
        self.assertEqual(self.read(detailed)[1], [])
        self.assertEqual(self.mix.read_bytes(), species_before)
        base_tree = self.root / "base.tre"
        base_tree.write_text("strain\t101\t1|10|101\ts1\t0\t0\t0\t4\t1\n")
        out_tree = self.root / "strain.tre"
        tree.run(str(base_tree), abundance, str(out_tree))
        self.assertEqual(out_tree.read_bytes(), b"")

    def test_empty_singleton_set_from_ambiguous_groups(self):
        input = self.table("ambiguous.tsv", rows=[[10, "G1", 100, 1, "10_1,10_2", "101,102", "s1,s2"]])
        selected = str(self.root / "topk.tsv")
        topk.run(input=input, out=selected, min_reads=5, topk=3)
        self.assertEqual(self.read(self.refine(selected)[1])[1], [])

    def test_dynamic_cutoff_can_remove_all_strains(self):
        input = self.table("input.tsv", rows=[[10, "G1", 6, 1, "10_1", 101, "s1"]])
        self.assertEqual(self.read(self.refine(input, threshold=30)[1])[1], [])

    def test_no_species_is_a_valid_empty_strain_outcome(self):
        self.mix.write_text("speciesID\tabundance\n")
        input = self.table("input.tsv", rows=[[10, "G1", 20, 1, "10_1", 101, "s1"]])
        self.assertEqual(self.read(self.refine(input)[1])[1], [])

    def test_legacy_empty_header_is_accepted(self):
        input = self.table("legacy.tsv", BASE)
        detailed, abundance = self.refine(input)
        self.assertTrue({"strain_taxid", "strain_name"}.issubset(self.read(detailed)[0]))
        self.assertEqual(self.read(abundance)[1], [])

    def test_missing_identity_columns_with_rows_still_fails(self):
        input = self.table("bad.tsv", BASE, [[10, "G1", 20, 1, "10_1"]])
        with self.assertRaisesRegex(SystemExit, "input missing columns"):
            self.refine(input)

    def test_missing_core_columns_even_when_empty_still_fails(self):
        input = self.table("bad.tsv", ["species", "strains"])
        with self.assertRaisesRegex(SystemExit, "input missing columns"):
            self.refine(input)

    def test_truly_empty_file_still_fails(self):
        input = self.root / "bad.tsv"
        input.write_text("")
        with self.assertRaisesRegex(SystemExit, "empty header"):
            self.refine(str(input))

    def test_normal_abundance_allocation_is_unchanged(self):
        input = self.table("input.tsv", rows=[
            [10, "G1", 30, 0.6, "10_1", 101, "s1"],
            [10, "G2", 20, 0.4, "10_2", 102, "s2"],
            [20, "G1", 10, 1, "20_1", 201, "s3"],
        ])
        _, abundance = self.refine(input)
        values = {r["strain_taxid"]: float(r["abundance"]) for r in self.read(abundance)[1]}
        self.assertEqual(values, {"101": 0.45, "102": 0.30, "201": 0.25})

    def test_empty_output_without_names_uses_two_columns(self):
        _, abundance = self.refine(self.table("input.tsv"), emit_name=False)
        self.assertEqual(self.read(abundance), (["strain_taxid", "abundance"], []))


if __name__ == "__main__":
    unittest.main()
