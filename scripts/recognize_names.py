'''Find text references to a person'''
import re
import json
import unittest

def find_mentions(text, person_info):
    '''Finds references to a person in text.'''
    person_mentioned = []
    for k, v in person_info.items():
        for iden in v['identifiers']:
            m = re.search(iden, text, re.IGNORECASE)
            if m: 
                # print(text[m.start():m.end()])
                person_mentioned.append(k)
                break # just need 1 match
    return person_mentioned


class TestFindMentions(unittest.TestCase):

    nelson = '''Floridians know they can count on Bill Nelson to deliver for
                our families. That's why he makes affordable healthcare a 
                priority.'''
    hawley = 'You just cant trust politician Josh Hawley.'
    random = '''Politicians (including your representative in the State 
                Senate) have cut over $4 billion from Arizona schools. Here's
                where the candidates in your district stand:'''
    
    with open("wmp/candidates_sample.json", "r") as f:
        cand_info = json.load(f)

    def test_find_mentions(self):
        self.assertEqual(
            find_mentions(self.nelson, self.cand_info), ['BILL_NELSON'])
        self.assertEqual(
            find_mentions(self.hawley, self.cand_info), ['JOSH_HAWLEY'])
        self.assertEqual(
            find_mentions(self.random, self.cand_info), [])
    

if __name__ == "__main__":
    unittest.main()