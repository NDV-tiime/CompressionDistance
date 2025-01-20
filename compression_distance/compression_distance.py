from pydivsufsort import lempel_ziv_factorization, longest_previous_factor


def LZ77(s: str) -> int:
    """
    Number of Lempel-Ziv-77 phrases obtained when compressing the string s
    Different from the Lempel-Ziv complexity.
    """
    lpf = longest_previous_factor(s)
    return len(lempel_ziv_factorization(lpf)) - 1


def compression_distance(source: str, target: str) -> int:
    """
    Implements the internal compression distance c_int from

    Ergun, Funda, S. Muthukrishnan, and S. Cenk Sahinalp.
    "Comparing sequences with segment rearrangements.", 2003.
    """
    source += chr(max(map(ord, source + target)) + 1)
    return LZ77(source + target) - LZ77(source)
