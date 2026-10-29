
import subprocess
import sys

steps = [
    ("Collect Apify posts", "src/collect_apify_tweets.py"),
    ("Analyze sentiment", "src/process_apify_tweets.py"),
]

for name, script in steps:
    print(f"\n{'=' * 50}")
    print(f"Step: {name}")
    print("=" * 50)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        raise SystemExit(
            f"Pipeline stopped because '{script}' failed."
        )

print("\nPipeline completed successfully.")
