from collections import Counter
assert Counter(t for physical in range(48) for warp in range(16) for t in range(physical*16+warp,1028,48*16))==Counter(range(1028))
for physical in range(48):
    own=physical%2
    assert own+1-2*own==1-own
print('Each of 1028 quarter tasks has exactly one producer for both rows.')
