import math
from functools import total_ordering


@total_ordering
class qROFN:
    """
    A class to represent a q-Rung Orthopair Fuzzy Number (q-ROFN).
    """

    def __init__(self, mu: float, nu: float, q: float) -> None:
        """
        Initializes the qROFN instance and validates the parameters.

        :param mu: Membership degree, must be in [0, 1]
        :param nu: Non-membership degree, must be in [0, 1]
        :param q: Rung parameter, must be >= 1
        """
        # 1. Validate domains for mu and nu
        if not (0.0 <= mu <= 1.0):
            raise ValueError(f"Membership degree 'mu' must be in [0, 1]. Got: {mu}")
        if not (0.0 <= nu <= 1.0):
            raise ValueError(f"Non-membership degree 'nu' must be in [0, 1]. Got: {nu}")

        # 2. Validate q parameter
        if q < 1.0:
            raise ValueError(f"Rung parameter 'q' must be >= 1. Got: {q}")

        # 3. Validate the core constraint: 0 <= mu^q + nu^q <= 1
        # Added a tiny tolerance (1e-9) to prevent floating-point strictness errors
        # when the sum is mathematically exactly 1 (e.g., 1.0000000000000002).
        sum_q = (mu ** q) + (nu ** q)
        if not (0.0 <= sum_q <= 1.0 + 1e-9):
            raise ValueError(
                f"Core constraint violated: 0 <= mu^q + nu^q <= 1. "
                f"Calculated mu^q + nu^q = {sum_q}"
            )

        self.mu = mu
        self.nu = nu
        self.q = q
        self.sum_q = sum_q


    @property
    def hesitancy(self) -> float:
        """
        Calculates the hesitancy degree (pi) of the qROFN.
        Formula: pi = (1 - mu^q - nu^q) ^ (1/q)
        """
        # Utilizing max(0.0, ...) to gracefully handle floating point inaccuracies
        # that might result in very small negative numbers instead of absolute 0.
        base_val = max(0.0, 1.0 - (self.mu ** self.q) - (self.nu ** self.q))
        return base_val ** (1.0 / self.q)

    def score(self) -> float:
        """
        Calculates the Score function (S) of the qROFN.
        Formula: S = mu^q - nu^q
        """
        return (self.mu ** self.q) - (self.nu ** self.q)

    def accuracy(self) -> float:
        """
        Calculates the Accuracy function (A) of the qROFN.
        Formula: A = mu^q + nu^q
        """
        return (self.mu ** self.q) + (self.nu ** self.q)

    def __eq__(self, other: object) -> bool:
        """
        Determines if two qROFNs are equal based on Score and Accuracy.
        """
        if not isinstance(other, qROFN):
            return NotImplemented

        # Check if Score AND Accuracy are equal (using float tolerance)
        return (math.isclose(self.score(), other.score(), abs_tol=1e-9) and
                math.isclose(self.accuracy(), other.accuracy(), abs_tol=1e-9))

    def __lt__(self, other: object) -> bool:
        """
        Determines if this qROFN is strictly less than another.
        """
        if not isinstance(other, qROFN):
            return NotImplemented

        s_self = self.score()
        s_other = other.score()

        # 1. Compare Scores first
        if not math.isclose(s_self, s_other, abs_tol=1e-9):
            return s_self < s_other

        # 2. If Scores are equal, compare Accuracies
        a_self = self.accuracy()
        a_other = other.accuracy()

        if not math.isclose(a_self, a_other, abs_tol=1e-9):
            return a_self < a_other

        # If both are fully equal, then self is NOT strictly less than other
        return False

    def __repr__(self) -> str:
        """String representation of the qROFN."""
        return f"qROFN(mu={self.mu}, nu={self.nu}, q={self.q})"


# ==========================================
# Testing and Demonstration Block
# ==========================================
if __name__ == "__main__":
    print("--- qROFN Instantiation and Validation ---")
    try:
        # Valid qROFN
        a1 = qROFN(mu=0.8, nu=0.5, q=3)
        print(f"Successfully created: {a1}")
        print(f"Hesitancy (pi) : {a1.hesitancy:.4f}")
        print(f"Score (S)      : {a1.score():.4f}")
        print(f"Accuracy (A)   : {a1.accuracy():.4f}\n")
    except ValueError as e:
        print(f"Error: {e}")

    print("--- Testing Constraints ---")
    try:
        # Invalid (mu^q + nu^q > 1 for q=1, as 0.8^1 + 0.5^1 = 1.3)
        invalid_a = qROFN(mu=0.8, nu=0.5, q=1)
    except ValueError as e:
        print(f"Constraint gracefully caught: {e}\n")

    print("--- Testing Comparisons ---")
    # Let's create a few numbers to test our comparison logic
    # a1: Score = (0.8^3 - 0.5^3) = 0.512 - 0.125 = 0.387
    # a2: Score = (0.7^3 - 0.2^3) = 0.343 - 0.008 = 0.335
    a2 = qROFN(mu=0.7, nu=0.2, q=3)

    print(f"a1: {a1} | Score: {a1.score():.3f} | Accuracy: {a1.accuracy():.3f}")
    print(f"a2: {a2} | Score: {a2.score():.3f} | Accuracy: {a2.accuracy():.3f}")
    print(f"a1 > a2  ? {a1 > a2}")  # Should be True (0.387 > 0.335)
    print(f"a1 < a2  ? {a1 < a2}\n")  # Should be False

    # Create variables with same scores but different accuracy
    # For a3 and a4, let's artificially construct them to have the exact same score.
    # We'll use values where mu^q - nu^q are equal but mu^q + nu^q differ.
    # a3 = (0.6, 0.4, 1) -> Score = 0.2, Accuracy = 1.0
    # a4 = (0.5, 0.3, 1) -> Score = 0.2, Accuracy = 0.8
    a3 = qROFN(mu=0.6, nu=0.4, q=1)
    a4 = qROFN(mu=0.5, nu=0.3, q=1)

    print("Testing Fallback to Accuracy:")
    print(f"a3: {a3} | Score: {a3.score():.3f} | Accuracy: {a3.accuracy():.3f}")
    print(f"a4: {a4} | Score: {a4.score():.3f} | Accuracy: {a4.accuracy():.3f}")
    print(f"a3 == a4 ? {a3 == a4}")  # Should be False
    print(f"a3 > a4  ? {a3 > a4}")  # Should be True (Because Accuracy 1.0 > 0.8)

    # Testing exact equality
    a5 = qROFN(mu=0.6, nu=0.4, q=1)
    print(f"\nTesting Equality (a3 == a5): {a3 == a5}")  # Should be True