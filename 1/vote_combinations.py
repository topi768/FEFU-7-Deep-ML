"""Независимая проверка n_min: ищет минимальное число голосов и сами комбинации оценок.

Не использует интервальную формулу. Для каждого n реально считает среднее S/n,
округляет его (2 знака, «половина вверх») и сравнивает с рейтингом, а затем
перебирает наборы оценок с такой суммой.
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from math import floor


def _to_r100(rating):
    """Рейтинг -> целое число сотых (4.87 -> 487), без ошибок float."""
    return int((Decimal(str(rating)) * 100).quantize(Decimal(1), ROUND_HALF_UP))


def _round100(s, n):
    """round(s/n, 2) в сотых, «половина вверх», точная арифметика."""
    return floor(Fraction(s, n) * 100 + Fraction(1, 2))


def _count_vectors(n, s, stars):
    """Все векторы (k_1..k_m): sum k = n, sum stars[i]*k_i = s, k_i >= 0."""
    m = len(stars)

    def rec(i, left_n, left_s):
        if i == m - 1:
            if left_n * stars[i] == left_s:
                yield (left_n,)
            return
        # границы на оставшуюся сумму: все оставшиеся голоса от min до max оценки
        for k in range(left_n + 1):
            rest_n, rest_s = left_n - k, left_s - k * stars[i]
            if rest_n * stars[i + 1] <= rest_s <= rest_n * stars[-1]:
                for tail in rec(i + 1, rest_n, rest_s):
                    yield (k,) + tail

    yield from rec(0, n, s)


def find_combinations(rating, lo=1, hi=5, max_combos=None, n_max=1000):
    """Минимальное n и комбинации оценок, дающие этот рейтинг.

    Возвращает (n, combos): combos это список словарей {оценка: сколько раз}.
    max_combos ограничивает число возвращаемых комбинаций (None = все).
    """
    stars = list(range(lo, hi + 1))
    target = _to_r100(rating)
    if not lo * 100 <= target <= hi * 100:
        raise ValueError(f"рейтинг {rating} вне диапазона [{lo}, {hi}]")
    for n in range(1, n_max + 1):
        combos = []
        for s in range(n * lo, n * hi + 1):
            if _round100(s, n) != target:
                continue
            for vec in _count_vectors(n, s, stars):
                combos.append({st: k for st, k in zip(stars, vec) if k})
                if max_combos is not None and len(combos) >= max_combos:
                    return n, combos
        if combos:
            return n, combos
    return None, []


if __name__ == "__main__":
    n, combos = find_combinations(4.96)
    print(n, len(combos))
    for c in combos[:5]:
        print(c)
