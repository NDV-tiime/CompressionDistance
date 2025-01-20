from compression_distance import compression_distance

def test_compression_distance():
    r = "vwxyz"
    q = "zyxwv"
    s = "wvxwyxzyyxwxwvzyxzyxwyxwvzyxwv"
    assert compression_distance(r, q) == 5
    assert compression_distance(q, s) == 10
    assert compression_distance(r, s) == 16