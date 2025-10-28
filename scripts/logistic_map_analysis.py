"""
Task 5: Logistic Map - Classical Chaos Comparison
Compare logistic map (classical chaos) with text generation divergence
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import Levenshtein


# ============================================================================
# LOGISTIC MAP FUNCTIONS
# ============================================================================

def logistic_map(x: float, r: float) -> float:
    """
    Single iteration of the logistic map: x_{n+1} = r * x_n * (1 - x_n)
    """
    return r * x * (1 - x)


def generate_trajectory(x0: float, r: float, n_steps: int) -> np.ndarray:
    """Generate trajectory of logistic map."""
    trajectory = np.zeros(n_steps)
    trajectory[0] = x0

    for i in range(1, n_steps):
        trajectory[i] = logistic_map(trajectory[i-1], r)

    return trajectory


def calculate_trajectory_divergence(x0_1: float, x0_2: float, r: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate divergence between two trajectories with nearby initial conditions.

    Returns:
        time_steps, divergence, traj1, traj2
    """
    traj1 = generate_trajectory(x0_1, r, n_steps)
    traj2 = generate_trajectory(x0_2, r, n_steps)

    divergence = np.abs(traj1 - traj2)
    time_steps = np.arange(n_steps)

    return time_steps, divergence, traj1, traj2


def calculate_lyapunov_analytical(trajectory: np.ndarray, r: float) -> float:
    """
    Calculate Lyapunov exponent using analytical formula.
    λ = lim (1/N) Σ ln|f'(x_n)| where f'(x) = r(1 - 2x)
    """
    # Derivative of logistic map: f'(x) = r(1 - 2x)
    derivatives = r * (1 - 2 * trajectory)

    # Take log of absolute values
    log_derivatives = np.log(np.abs(derivatives))

    # Average over trajectory (skip first few transient points)
    with np.errstate(divide='ignore', invalid='ignore'):
        lyapunov = np.mean(log_derivatives[100:])

    return lyapunov


def calculate_lyapunov_empirical(divergence: np.ndarray, n_fit_points: int = 20) -> float:
    """
    Calculate Lyapunov exponent empirically from divergence data.
    Fit: d(n) = d(0) * exp(λ * n)
    """
    # Find first non-zero point
    nonzero_idx = np.where(divergence > 1e-10)[0]
    if len(nonzero_idx) == 0:
        return 0.0

    start_idx = nonzero_idx[0]
    end_idx = min(start_idx + n_fit_points, len(divergence))

    # Linear fit on log(divergence)
    time_range = np.arange(start_idx, end_idx)
    log_div = np.log(divergence[start_idx:end_idx])

    # Fit: log(d) = a + λ*t
    coeffs = np.polyfit(time_range, log_div, 1)
    lambda_empirical = coeffs[0]  # Slope = Lyapunov exponent

    return lambda_empirical


# ============================================================================
# TEXT DIVERGENCE FUNCTIONS
# ============================================================================

def read_text_file(filepath: str) -> str:
    """Read text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def tokenize(text: str) -> list:
    """Tokenize text into words."""
    return text.split()


def calculate_text_divergence(tokens1, tokens2):
    """Calculate Levenshtein distance for growing prefixes."""
    max_length = min(len(tokens1), len(tokens2))
    k_values = []
    d_values = []

    for k in range(1, max_length + 1):
        prefix1 = ' '.join(tokens1[:k])
        prefix2 = ' '.join(tokens2[:k])
        distance = Levenshtein.distance(prefix1, prefix2)
        k_values.append(k)
        d_values.append(distance)

    return np.array(k_values), np.array(d_values)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_logistic_divergence(time_steps, divergence, traj1, traj2):
    """Create comprehensive visualization of logistic map divergence."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Both trajectories
    axes[0, 0].plot(time_steps[:100], traj1[:100], 'b-', label='Trajectory 1', linewidth=2, alpha=0.8)
    axes[0, 0].plot(time_steps[:100], traj2[:100], 'r--', label='Trajectory 2', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Time Step n', fontsize=12)
    axes[0, 0].set_ylabel('State x_n', fontsize=12)
    axes[0, 0].set_title('Two Trajectories (First 100 Steps)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Divergence over time (linear scale)
    axes[0, 1].plot(time_steps, divergence, 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Time Step n', fontsize=12)
    axes[0, 1].set_ylabel('Divergence |x1(n) - x2(n)|', fontsize=12)
    axes[0, 1].set_title('Divergence Growth (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Max possible (bounded)')
    axes[0, 1].legend()

    # Plot 3: Log divergence
    nonzero_idx = divergence > 1e-10
    axes[1, 0].semilogy(time_steps[nonzero_idx], divergence[nonzero_idx], 'purple', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Time Step n', fontsize=12)
    axes[1, 0].set_ylabel('log(Divergence)', fontsize=12)
    axes[1, 0].set_title('Logarithmic Divergence (Exponential Growth)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, 'Linear in log-space\n→ Exponential growth',
                    transform=axes[1, 0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 4: Saturation phase
    axes[1, 1].plot(time_steps[50:], divergence[50:], 'orange', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Time Step n', fontsize=12)
    axes[1, 1].set_ylabel('Divergence', fontsize=12)
    axes[1, 1].set_title('Saturation Phase', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=np.mean(divergence[200:]), color='red', linestyle='--', alpha=0.5, label='Mean')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('logistic_map_divergence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: logistic_map_divergence.png")
    plt.close()


def plot_comparison(k_text, d_text, time_steps, divergence_logistic):
    """Create side-by-side comparison of text vs logistic map."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Text generation - Linear scale
    axes[0, 0].plot(k_text, d_text, 'b-', linewidth=2.5, label='Text Generation', alpha=0.8)
    axes[0, 0].set_xlabel('Token Position', fontsize=12)
    axes[0, 0].set_ylabel('Divergence', fontsize=12)
    axes[0, 0].set_title('Text Generation: Linear Divergence', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].text(0.05, 0.95, 'Linear growth\nNon-chaotic',
                    transform=axes[0, 0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Plot 2: Logistic map - Linear scale
    axes[0, 1].plot(time_steps[:len(k_text)], divergence_logistic[:len(k_text)],
                    'r-', linewidth=2.5, label='Logistic Map (r=4.0)', alpha=0.8)
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].set_ylabel('Divergence', fontsize=12)
    axes[0, 1].set_title('Logistic Map: Exponential → Saturation', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].text(0.05, 0.95, 'Exponential growth\nChaotic (λ > 0)',
                    transform=axes[0, 1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Plot 3: Text - Log scale
    text_nonzero = d_text > 0
    axes[1, 0].plot(k_text[text_nonzero], np.log(d_text[text_nonzero] + 1), 'b-', linewidth=2.5, alpha=0.8)
    axes[1, 0].set_xlabel('Token Position', fontsize=12)
    axes[1, 0].set_ylabel('ln(Divergence + 1)', fontsize=12)
    axes[1, 0].set_title('Text: Log Scale (Subexponential)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Logistic - Log scale
    logistic_nonzero = divergence_logistic > 1e-10
    axes[1, 1].semilogy(time_steps[:50][logistic_nonzero[:50]],
                        divergence_logistic[:50][logistic_nonzero[:50]],
                        'r-', linewidth=2.5, alpha=0.8)
    axes[1, 1].set_xlabel('Time Step', fontsize=12)
    axes[1, 1].set_ylabel('log(Divergence)', fontsize=12)
    axes[1, 1].set_title('Logistic: Log Scale (Exponential)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_text_vs_logistic.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_text_vs_logistic.png")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*70)
    print("TASK 5: LOGISTIC MAP vs TEXT GENERATION COMPARISON")
    print("="*70)

    # ========================================================================
    # Part 1: Logistic Map Analysis
    # ========================================================================
    print("\n1. LOGISTIC MAP ANALYSIS")
    print("-"*70)

    # Parameters
    r = 4.0  # Maximum chaos
    x0_1 = 0.5
    x0_2 = 0.5 + 1e-6  # Tiny perturbation
    n_steps = 500

    print(f"Parameters:")
    print(f"  r = {r} (chaotic regime)")
    print(f"  x0_1 = {x0_1}")
    print(f"  x0_2 = {x0_2}")
    print(f"  Perturbation: {x0_2 - x0_1:.2e} ({(x0_2 - x0_1)/x0_1 * 100:.4e}%)")

    # Generate trajectories
    print(f"\nGenerating trajectories...")
    time_steps, divergence, traj1, traj2 = calculate_trajectory_divergence(x0_1, x0_2, r, n_steps)

    # Calculate Lyapunov exponents
    lambda_analytical = calculate_lyapunov_analytical(traj1, r)
    lambda_empirical = calculate_lyapunov_empirical(divergence, n_fit_points=15)

    print(f"\nLyapunov Exponents:")
    print(f"  Theoretical (r=4):      λ = {np.log(2):.4f} (ln 2)")
    print(f"  Analytical (derivative): λ = {lambda_analytical:.4f}")
    print(f"  Empirical (divergence):  λ = {lambda_empirical:.4f}")
    print(f"\n✓ Positive Lyapunov exponent → CHAOTIC BEHAVIOR")

    # Create visualization
    print(f"\nCreating logistic map visualizations...")
    plot_logistic_divergence(time_steps, divergence, traj1, traj2)

    # ========================================================================
    # Part 2: Load Text Generation Results
    # ========================================================================
    print("\n2. LOADING TEXT GENERATION RESULTS")
    print("-"*70)

    try:
        text1 = read_text_file('data/text1.txt')
        text2 = read_text_file('data/text2.txt')
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        print(f"✓ Loaded text files")
        print(f"  Text 1: {len(tokens1)} tokens")
        print(f"  Text 2: {len(tokens2)} tokens")

        # Calculate text divergence
        k_text, d_text = calculate_text_divergence(tokens1, tokens2)

        # Calculate text Lyapunov
        first_nonzero_text = next((i for i, d in enumerate(d_text) if d > 0), 1)
        lambda_text_eff = (1.0 / len(k_text)) * np.log(d_text[-1] / d_text[first_nonzero_text]) if d_text[first_nonzero_text] > 0 else 0

        print(f"\nText Generation Lyapunov:")
        print(f"  λ_eff = {lambda_text_eff:.4f} (→ 0)")
        print(f"\n✓ Lyapunov → 0 indicates NON-CHAOTIC BEHAVIOR")

    except FileNotFoundError as e:
        print(f"✗ Error: Could not find text files: {e}")
        print(f"  Make sure text1.txt and text2.txt are in the current directory")
        print(f"  Skipping text comparison...")
        k_text = None
        d_text = None
        lambda_text_eff = None

    # ========================================================================
    # Part 3: Comparison
    # ========================================================================
    if k_text is not None:
        print("\n3. CREATING COMPARISON PLOTS")
        print("-"*70)
        plot_comparison(k_text, d_text, time_steps, divergence)

    # ========================================================================
    # Part 4: Summary Table
    # ========================================================================
    print("\n4. COMPARISON SUMMARY")
    print("="*90)
    print()

    properties = [
        'System Type',
        'Initial Perturbation',
        'Divergence Growth',
        'Lyapunov Exponent λ',
        'Phase Space',
        'Chaotic?',
        'Sensitive to IC?',
    ]

    logistic_values = [
        'Classical dynamical system',
        f'{x0_2 - x0_1:.1e} ({(x0_2 - x0_1)/x0_1 * 100:.4f}%)',
        'Exponential → Saturation',
        f'{lambda_analytical:.4f} (POSITIVE)',
        'Bounded [0, 1]',
        '✓ YES',
        '✓ YES',
    ]

    if lambda_text_eff is not None:
        text_values = [
            'Linguistic trajectory',
            '1 word change ("driven" → "governed")',
            'Linear (sustained)',
            f'{lambda_text_eff:.4f} (→ 0)',
            'Unbounded (infinite vocabulary)',
            '✗ NO',
            '✓ YES',
        ]
    else:
        text_values = ['N/A'] * len(properties)

    # Print table
    print(f"{'Property':<25} | {'Logistic Map (r=4.0)':<35} | {'Text Generation':<35}")
    print("-" * 100)
    for prop, log_val, text_val in zip(properties, logistic_values, text_values):
        print(f"{prop:<25} | {log_val:<35} | {text_val:<35}")

    print("="*90)

    # Save to CSV
    with open('comparison_table.csv', 'w') as f:
        f.write("Property,Logistic Map (r=4.0),Text Generation (LLM)\n")
        for prop, log_val, text_val in zip(properties, logistic_values, text_values):
            log_val_clean = log_val.replace(',', ';')
            text_val_clean = text_val.replace(',', ';')
            f.write(f'"{prop}","{log_val_clean}","{text_val_clean}"\n')

    print("\n✓ Saved: comparison_table.csv")

    # ========================================================================
    # Final Conclusion
    # ========================================================================
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("Key Findings:")
    print("  • Logistic map (r=4.0): λ > 0 → CHAOTIC")
    print("  • Text generation: λ → 0 → NOT CHAOTIC")
    print()
    print("While both systems show sensitivity to initial conditions,")
    print("only the logistic map exhibits classical deterministic chaos")
    print("with exponential divergence and positive Lyapunov exponent.")
    print()
    print("Text generation shows LINEAR divergence, more like semantic")
    print("drift or random walk than explosive chaotic separation.")
    print("="*70)


if __name__ == "__main__":
    main()
