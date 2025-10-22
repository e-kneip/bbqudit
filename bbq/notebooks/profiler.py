import numpy as np


def main():
    p = 3
    order = 1
    dim = 20

    guess = np.zeros(dim, dtype=int)
    for decimal in range(2**dim):
        index = 0
        for i in range(dim):
            val = decimal // (2**i) % 2
            guess[i] = val
            if val:
                index += 1
        if index > order:
            continue  # ideally would break out of two loops here :(
        print(guess)

        for i in range(2, p):
            guess = i*np.array(guess)
            print(guess)
            guess = guess // i


if __name__ == "__main__":
    main()
