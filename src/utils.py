def u64_to_bits(n):
    if n < 0 or n > 2 ** 64 - 1:
        raise ValueError("Not u64")

    B = []
    for _ in range(64):
        b = n % 2
        n //= 2
        B.append(b)
    return list(reversed(B))


def u64_from_bits(B):
    if len(B) != 64:
        raise ValueError("Not u64")

    n = 0
    for i in range(64):
        n += int(B[i]) * 2 ** (63 - i)
    return n
