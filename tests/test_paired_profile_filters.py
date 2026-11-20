import tempfile
import unittest
from pathlib import Path

from pantax_dbg.profile import _finalize_paired_profiles


class PairedProfileFilterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.work = self.out / "work"
        self.work.mkdir()
        self.prefix = self.work / "query"
        self.reads = self.work / "reads.fastq"
        self.counts = self.work / "query.species_counts.tsv"
        self.map = self.work / "mapping.tsv"
        self.ganon = self.work / "ganon.tsv"
        self.tree = self.work / "ganon.tre"

        genome = self.work / "ref.fasta"
        genome.write_text(">ref\nACGT\n")
        self.map.write_text(
            "species_taxid\tstrain_taxid\tstrain_name\tnew_strain_taxid\tgenome_path\n"
            f"10\t101\ts1\t10_1\t{genome}\n"
            f"20\t201\ts2\t20_1\t{genome}\n"
        )
        self.ganon.write_text("10\t0.6\n20\t0.4\n")
        self.tree.write_text(
            "root\t1\t1\troot\t0\t0\t21\t21\t100\n"
            "species\t10\t1|10\tsp1\t0\t0\t11\t11\t60\n"
            "strain\t101\t1|10|101\ts1\t11\t0\t0\t11\t60\n"
            "species\t20\t1|20\tsp2\t0\t0\t10\t10\t40\n"
            "strain\t201\t1|20|201\ts2\t10\t0\t0\t10\t40\n"
        )
        (self.work / "query.strain_group_abundance.tsv").write_text(
            "species\tgroup_id\tassigned_reads\trel_abundance_within_species\tstrains\n"
            "10\tG1\t10\t1\t10_1\n"
            "20\tG1\t10\t1\t20_1\n"
        )

    def profile(self, counts, total_reads, *, species_min_reads=0.0, strain_min_reads=5.0):
        self.counts.write_text(counts)
        self.reads.write_text("@read\nACGT\n+\nIIII\n" * total_reads)
        return _finalize_paired_profiles(
            out_prefix=self.out,
            work_q=self.work,
            ggcat_prefix=str(self.prefix),
            tmp_id_table=str(self.map),
            ganon_predict_spy=str(self.ganon),
            tre_file=str(self.tree),
            reads_path=self.reads,
            species_min_reads=species_min_reads,
            species_min_abundance=0.0,
            strain_min_reads=strain_min_reads,
            strain_topk=5,
            strain_div=5.0,
        )

    def assert_empty_profiles(self):
        self.assertEqual((self.out / "species_abundance.txt").read_text(), "speciesID\tabundance\n")
        self.assertEqual((self.out / "strain_abundance.txt").read_text(), "strain_taxid\tstrain_name\tabundance\n")
        self.assertEqual((self.out / "tax_profile.tre").read_text(), "")
        self.assertEqual((self.out / "tax_profile_strain.tre").read_text(), "")

    def test_low_dynamic_threshold_clears_profiles(self):
        self.profile("10\t10\n", 2000)
        self.assert_empty_profiles()

    def test_threshold_equal_to_boundary_can_report_strains(self):
        self.profile("10\t10\n", 1000)
        self.assertIn("10\t1.0\n", (self.out / "species_abundance.txt").read_text())
        self.assertIn("101\ts1\t1.0000000000\n", (self.out / "strain_abundance.txt").read_text())

    def test_species_support_must_exceed_minimum_and_profiles_are_renormalized(self):
        self.profile("10\t11\n20\t10\n", 100, species_min_reads=10)
        self.assertEqual((self.out / "species_abundance.txt").read_text(), "speciesID\tabundance\n10\t1.0\n")
        self.assertEqual(
            (self.out / "strain_abundance.txt").read_text(),
            "strain_taxid\tstrain_name\tabundance\n101\ts1\t1.0000000000\n",
        )
        tree = (self.out / "tax_profile.tre").read_text().splitlines()
        self.assertEqual([line.split("\t")[1] for line in tree], ["1", "10"])
        self.assertTrue(all(len(line.split("\t")) == 9 for line in tree))

    def test_no_reportable_strains_overwrites_previous_positive_profiles(self):
        self.profile("10\t11\n20\t10\n", 100)
        self.assertIn("201\ts2\t", (self.out / "strain_abundance.txt").read_text())
        self.profile("10\t11\n20\t10\n", 100, strain_min_reads=10)
        self.assert_empty_profiles()

    def test_all_species_below_minimum_produces_empty_profiles(self):
        self.profile("10\t11\n20\t10\n", 100, species_min_reads=11)
        self.assert_empty_profiles()
