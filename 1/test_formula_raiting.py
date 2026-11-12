CONST_RATING = 4.99
def n_min_simple(r, n_max=10_000):
    r100 = round(r * 100)
    for n in range(1, n_max + 1):
        s = -(-n * (2 * r100 - 1) // 200)      # ceil(n*(r-0.005))
        if 200 * s < n * (2 * r100 + 1):        # s/n < r+0.005
            return n
    return None
print(n_min_simple(CONST_RATING))



from vote_combinations import find_combinations

print(find_combinations(CONST_RATING))


