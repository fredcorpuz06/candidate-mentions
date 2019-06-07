import unittest
import wiki_image

class TestCalc(unittest.TestCase):

    def test_clean_state(self):
        self.assertEqual(
            wiki_image.clean_state("Minnesota special Election"),
            "Minnesota")
        self.assertEqual(
            wiki_image.clean_state("Montana election"),
            "Montana")
    def test_clean_cand(self):
        self.assertEqual(
            wiki_image.clean_cand("Kevin de León"), "Kevin_de_Leon")
        self.assertEqual(
            wiki_image.clean_cand("Beto O'Rourke"), "Beto_O_Rourke")
        self.assertEqual(
            wiki_image.clean_cand("Bob Casey Jr."), "Bob_Casey_Jr")

if __name__ == "__main__":
    unittest.main()