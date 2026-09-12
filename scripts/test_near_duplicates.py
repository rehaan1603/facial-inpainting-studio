import unittest
import random
from audit_near_duplicates import candidates


class HashSearch(unittest.TestCase):
    def test_exact_against_bruteforce(self):
        rng=random.Random(91);values=[rng.getrandbits(64) for _ in range(100)]
        for n in range(7):
            changed=values[0]
            for bit in rng.sample(range(64),n):changed^=1<<bit
            values.append(changed)
        actual=set(candidates(values));expected={(i,j,(a^b).bit_count()) for i,a in enumerate(values) for j,b in enumerate(values) if i<j and (a^b).bit_count()<=6}
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
