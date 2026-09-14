"""
real_dataset_check.py

Downloads the real LLMDrift dataset (Chen, Zaharia, Zou 2023) and runs it
through the ACTUAL calibration.py + bootstrap.py modules to confirm the
detection pipeline correctly identifies the real, documented GPT-4
March-2023-vs-June-2023 capability regression on the "is this a prime
number" task.

No API calls, no cost -- this is real historical closed-model data,
released by the original paper's authors under Apache-2.0.
"""

import re
import urllib.request
import statistics
from pathlib import Path

import pandas as pd

from driftguard.detection import calibration as cal
from driftguard.detection import bootstrap as bs

CSV_URL = "https://raw.githubusercontent.com/lchen001/LLMDrift/main/generation/PRIME_FULL_EVAL.csv"
LOCAL_PATH = Path("llmdrift_prime_full_eval.csv")

BATCH_SIZE = 40  # bucket the 1000 queries per snapshot into batches, to get a
                  # real per-batch accuracy series (mirrors how our own eval
                  # suite produces one accuracy number per RUN, not per query)


def download_if_needed():
    if not LOCAL_PATH.exists():
        print(f"Downloading real dataset from {CSV_URL} ...")
        urllib.request.urlretrieve(CSV_URL, LOCAL_PATH)
        print(f"Saved to {LOCAL_PATH}")
    else:
        print(f"Using cached {LOCAL_PATH}")


def extract_yes_no(answer_text: str) -> str:
    """Real-world answer parsing -- same spirit as our own _extract_answer_letter
    in tasks.py, adapted to this dataset's yes/no + bracket format."""
    if not isinstance(answer_text, str):
        return None
    text = answer_text.lower()
    if "[yes]" in text:
        return "yes"
    if "[no]" in text:
        return "no"
    # fallback: bare yes/no with no brackets (seen in some June responses)
    stripped = text.strip()
    if stripped.startswith("yes"):
        return "yes"
    if stripped.startswith("no"):
        return "no"
    return None


def batch_accuracy(df_subset: pd.DataFrame, batch_size: int) -> list:
    """Split a model's queries into sequential batches, compute accuracy per batch."""
    df_subset = df_subset.reset_index(drop=True)
    accuracies = []
    for start in range(0, len(df_subset), batch_size):
        batch = df_subset.iloc[start:start + batch_size]
        if len(batch) < batch_size:
            continue  # drop a trailing partial batch, keep batches comparable size
        correct = 0
        for _, row in batch.iterrows():
            predicted = extract_yes_no(row["answer"])
            actual = str(row["ref_answer"]).strip().lower()
            if predicted == actual:
                correct += 1
        accuracies.append(correct / len(batch))
    return accuracies


def main():
    download_if_needed()
    df = pd.read_csv(LOCAL_PATH)

    march = df[df["model"] == "openaichat/gpt-4-0314"]
    june = df[df["model"] == "openaichat/gpt-4-0613"]

    print(f"\nMarch 2023 GPT-4 queries: {len(march)}")
    print(f"June 2023 GPT-4 queries: {len(june)}")

    march_batches = batch_accuracy(march, BATCH_SIZE)
    june_batches = batch_accuracy(june, BATCH_SIZE)

    print(f"\nMarch per-batch accuracies ({len(march_batches)} batches of {BATCH_SIZE}):")
    print("  ", [round(a, 3) for a in march_batches])
    print(f"Overall March accuracy: {statistics.mean(march_batches):.3f}")

    print(f"\nJune per-batch accuracies ({len(june_batches)} batches of {BATCH_SIZE}):")
    print("  ", [round(a, 3) for a in june_batches])
    print(f"Overall June accuracy: {statistics.mean(june_batches):.3f}")

    print("\n(Paper's reported headline numbers: March 84.0%, June 51.1% -- our")
    print(" independent re-grading of the raw released responses should be in the same ballpark.)")

    print("\n" + "=" * 70)
    print("Running through calibration.py: noise profile from March batches")
    print("=" * 70)
    profile = cal.compute_noise_profile("accuracy", march_batches)
    print(f"  median={profile.median:.4f}  mad_scaled={profile.mad_scaled:.4f}  "
          f"mean={profile.mean:.4f}  std={profile.std:.4f}")

    print("\n" + "=" * 70)
    print("Running through bootstrap.py: is March-vs-June a REAL, significant shift?")
    print("=" * 70)
    observed_delta, p_value = bs.permutation_test(
        pre_values=march_batches,
        post_values=june_batches,
        direction="lower_is_bad",
        n_permutations=5000,
    )
    print(f"  Observed delta (June - March): {observed_delta:.4f}")
    print(f"  p-value: {p_value:.5f}")
    print(f"  Significant at alpha=0.05: {p_value < 0.05}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    if p_value < 0.05 and observed_delta < 0:
        print("Our detection pipeline correctly identifies this REAL, documented")
        print("GPT-4 capability regression as statistically significant drift,")
        print("using the actual bootstrap.py significance test built for this project.")
    else:
        print("Did not confirm significance -- check batch sizes / parsing logic.")


if __name__ == "__main__":
    main()