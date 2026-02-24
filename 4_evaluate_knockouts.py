from general_utils import *
from knockout_utils import *
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run knockouts on LLAVA, Qwen, and Janus models for color or counting tasks.")
    parser.add_argument('--model_version', type=str, choices=[ 'qwen','llava-one', 'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=[ 'count', 'color'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--eval_type', type=str, choices=[ 'best', 'all'], required=True, help="Choose whether you're evaluating knockout or running the full sweep.")
    #If doing full sweep, evaluation = False, use the following:
    parser.add_argument( "--layer_lower", type=int, default=None, help="Lower bound (inclusive) for layer index")
    parser.add_argument("--layer_upper", type=int, default=None, help="Upper bound (exclusive) for layer index")
    parser.add_argument( "--head_lower", type=int, default=None, help="Lower bound (inclusive) for layer index")
    parser.add_argument("--head_upper", type=int, default=None, help="Upper bound (exclusive) for layer index")
    #If doing only doing evaluation, evaluation = True, use the following:
    parser.add_argument("--heads_to_ablate", type=str, default=None, help="Top 1/2/3.")
    
    args = parser.parse_args()

    if args.eval_type == "best":
        df = pd.read_csv(f"best_PIH_ablation_results_L{args.layer_lower}H{args.head_lower}_to_L{args.layer_upper}H{args.head_upper}_{args.model_version}_{args.task}_{args.dataset_size}.csv") 
    else:
        df = pd.read_csv(f"all_PIH_ablation_results_L{args.layer_lower}H{args.head_lower}_to_L{args.layer_upper}H{args.head_upper}_{args.model_version}_{args.task}_{args.dataset_size}.csv") 

    if args.model_version == "llava-one":
        df["ablated_text"] = df["ablated_text"].str.split(".assistant\n").str[1]
        df["baseline_text"] = df["baseline_text"].str.split(".assistant\n").str[1]


    if args.task == "count":
        df["category"] = df.apply(
            lambda row: categorize_change_with_nums(
                baseline_text=row["baseline_text"],
                ablated_text=row["ablated_text"],
                model_version=args.model_version,         
                ground_truth=row["ground_truth"]
            ),
            axis=1
        )
        
        correct = df[df["category"] == "number changed to ground truth"]
        if args.eval_type == "best":
            correct.to_csv(f"correct_{args.model_version}_{args.task}_{args.dataset_size}.csv") 
        else: 
            total_counts = (
                df
                .groupby(["layer", "head", "ground_truth"])
                .size()
                .reset_index(name="total_count")
            )

            # --- 2. Correct count per (layer, head, ground_truth) ---
            correct_counts = (
                correct
                .groupby(["layer", "head", "ground_truth"])
                .size()
                .reset_index(name="correct_count")
            )

            # --- 3. Merge and compute percentage accuracy ---
            merged = (
                pd.merge(
                    correct_counts,
                    total_counts,
                    on=["layer", "head", "ground_truth"],
                    how="left",
                )
            )

            merged["percent"] = 100 * merged["correct_count"] / merged["total_count"]

            # --- 4. Build head identifier (e.g., L0H3) ---
            merged["head_id"] = (
                "L" + merged["layer"].astype(int).astype(str)
                + "H" + merged["head"].astype(int).astype(str)
            )

            for k in [5, 4, 3, 2, 1]:
                heads = collect_top_heads_with_ties(merged, k)
                print(f"\nTop {k}:")
                print(heads)
                print(len(heads))

            DATASET_SIZE = len(df[(df["layer"] == 0) & (df["head"] == 0)])

            # Aggregate correct counts across all ground truths
            overall = (
                merged
                .groupby(["layer", "head"], as_index=False)
                .agg(correct_count=("correct_count", "sum"))
            )

            # Use full dataset size as denominator
            overall["total_count"] = DATASET_SIZE
            overall["percent"] = 100 * overall["correct_count"] / DATASET_SIZE

            # Head identifier (e.g., L0H3)
            overall["head_id"] = (
                "L" + overall["layer"].astype(int).astype(str)
                + "H" + overall["head"].astype(int).astype(str)
            )

            # Rank heads by overall accuracy (tie-safe)
            TOP_K = 10
            overall["rank"] = overall["percent"].rank(method="dense", ascending=False)

            best_heads_overall = (
                overall[overall["rank"] <= TOP_K]
                .sort_values(["rank", "percent"], ascending=[True, False])
                .reset_index(drop=True)
            )

            best_heads_overall[["rank", "head_id", "percent", "correct_count", "total_count"]]
            best_heads_overall.to_csv(f"best_heads_overall_{args.model_version}_{args.task}_{args.dataset_size}.csv")

    else:
        #for color, we only use the "best" heads found in counts. We don't re-discover heads and do full knockouts. 
        df['cb_entity'] = df['prompt'].map(extract_entity_color) #df['first_wrong_prompt'].map(extract_entity)

        df["cb"] = df["cb_entity"].str.split(" ").str[0]

        df["entity"] = df["cb_entity"].str.split(" ").str[1:].str.join(" ")
        
        df["gt_entity"] =df["ground_truth"] + " " + df["entity"]

        df["category"] = df.apply(categorize_row_color, axis=1)
        df = df[df["ground_truth"] != "purple"]
        df = df[df["category"] != "truncation_unclear"]
        correct = df[df["category"].isin(["not_cb","gt_entity_mentioned", "gt_color"])]
        correct.to_csv(f"correct_{args.model_version}_{args.task}_{args.dataset_size}.csv") 




if __name__ == "__main__":
    main()
