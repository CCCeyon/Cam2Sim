import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import OUTPUT_FOLDER_NAME
from utils.argparser import parse_figure_args

# Load your CSV
args = parse_figure_args()

if args.name is None:
    args.name = args.value

if args.pgf is None:
    args.pgf = "figure"

file = os.path.join(OUTPUT_FOLDER_NAME,args.output_dir, args.file)
df = pd.read_csv(file, sep=",")  # adjust sep if needed


df.columns = df.columns.str.strip()

print("Detected columns:", df.columns.tolist())

extra = ""
extra2 = "no"
if args.no_seed:
    extra = "no"
if args.rotate:
    extra2 = ""

def parse_version(v):
    # ends with or is exactly carla
    if v.endswith("/carla") or v == "carla":
        return 0, "baseline"
    # Example: output/z_valtest_real/output-10_g3.5_seed_norotate
    match = re.search(r"output-(\d+)_g([\d.]+)_"+extra+"seed_"+extra2+"rotate", v)
    if match:
        split = int(match.group(1))
        g = float(match.group(2))
        return split, g
    print("no match for version:", v)
    return None, None

# Filter for seed_norotate versions
if args.file == "veh_results.csv":
    df = df[df["Version"].str.contains("_"+extra+"seed_"+extra2+"rotate")]

    df[["Split", "g"]] = df["Version"].apply(parse_version).apply(pd.Series)
elif args.file == "distribution_results.csv":
    df = df[df["name"].str.contains("_" + extra + "seed_"+extra2+"rotate")]
    df[["Split", "g"]] = df["name"].apply(lambda v: pd.Series(parse_version(v)))
elif args.file == "distribution.csv" or args.file == "results.csv" or args.file == "results_tempconsistency.csv":
    if args.file != "distribution.csv":
        df = df[df["Type"].str.contains("AVERAGE")]
    df = df[df["Folder-Name"].str.contains("_"+extra+"seed_"+extra2+"rotate") | df["Folder-Name"].str.contains("/carla") | (df["Folder-Name"] == "carla")]

    df[["Split", "g"]] = df["Folder-Name"].apply(lambda v: pd.Series(parse_version(v)))

    baseline_rows = df[df["g"] == "baseline"]

    if not baseline_rows.empty:
        carla_row = baseline_rows.iloc[0]

        # generate splits 0, 10, 20, ..., 100
        splits = list(range(0, 101, 10))

        # expand the single row into many rows
        baseline_df = pd.DataFrame([carla_row] * len(splits))
        baseline_df["Split"] = splits

        # concat back to the main df
        df = pd.concat([df, baseline_df], ignore_index=True)

# Extract split and g values from the Version string

# Sort for nice plotting
df = df.sort_values(by=["g", "Split"])

# Plot
#plt.figure(figsize=(7,5))
plt.figure(figsize=(5.6,4))

#plt.rcParams.update({
#    "font.size": 14,           # Default font size
#})

for g, sub in df.groupby("g"):
    if g == "baseline":
        plt.plot(
            sub["Split"], sub[args.value],
            marker="o", color="black", linestyle="--", label="Target"
        )
    else:
        plt.plot(
            sub["Split"], sub[args.value],
            marker="o", label=f"g={g}"
        )
plt.ylabel(args.name)
plt.xlabel("Split Position")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.title(
    f"{args.name} vs Split ("+ ("set" if not args.no_seed else "random") +" seed, cond. order "+ ("B" if args.rotate else "A")+")")
plt.legend()
plt.grid(True)

# Save LaTeX PGFPlots figure
plt.savefig(f"{args.pgf}.pgf")

# Also save a PNG for quick checking
plt.savefig("figure.png", dpi=300)

print("✅ Saved figure.pgf (LaTeX) and figure.png (preview)")
