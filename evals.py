import argparse
import pandas as pd
import regex as re
import time
from tqdm import tqdm
import evaluate
import logging
import sys

from create_agent import run_agent
from utils.load_graph import load_graph   

EVENT_TYPES = (
    "ACCUSE", "ASSAULT", "AID", "REQUEST", "PROTEST", "COERCE", "THREATEN",
    "RETREAT", "MOBILIZE", "SANCTION", "CONCEDE", "COOPERATE",
    "CONSULT", "REJECT"
)

# ------------- LOGGING SETUP -------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def extract_pred(answer_text):
    parts = [p.strip().upper() for p in answer_text.split(",") if p.strip()]
    valid = [p for p in parts if p in EVENT_TYPES]
    return (valid + [""]*3)[:3]

def get_predictions(graph, query, actor, recipient, date, debug=False):
    if debug:
        logging.info(f"Query: {query}")
    try:
        result = run_agent(
            graph=graph,
            query=query,
            actor=actor,
            recipient=recipient,
            date=date
        )
    except Exception as e:
        logging.error(f"Error running agent for query: {query} | Exception: {e}", exc_info=True)
        return ["", "", ""], [], str(e)

    last = result["messages"][-1].content
    all_messages_str = "\n\n".join(
        f"{getattr(m, 'role', type(m).__name__)}: {getattr(m, 'content', str(m))}"
        for m in result["messages"]
    )

    m = re.search(r"Answer:\s*(.*)", last, re.IGNORECASE)
    if not m:
        return ["", "", ""], result["messages"], all_messages_str
    preds = extract_pred(m.group(1))
    return preds, result["messages"], all_messages_str

def evaluate_agent(graph_pkl, input_csv, output_csv=None, sleep_time=1.0, debug=False):
    graph = load_graph(graph_pkl)
    df = pd.read_csv(input_csv)
    refs = df["Event Type"].astype(str).str.upper().tolist()

    preds1, preds2, preds3 = [], [], []
    raw_outputs = []
    all_messages_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        a = row["Actor Name"]
        r = row["Recipient Name"]
        d = row["Event Date"]

        # Log all input fields for each row (with repr to catch types)
        logging.info(f"Row {idx}: Actor={a!r}, Recipient={r!r}, Date={d!r}")

        # Defensive: convert to string to avoid NaN/float issues
        a_str = str(a) if pd.notna(a) else ""
        r_str = str(r) if pd.notna(r) else ""
        d_str = str(d) if pd.notna(d) else ""

        query_str = (
            f"What is the most likely relation between '{a_str}' and '{r_str}' "
            f"on {d_str}? Pick your top 3 from: {EVENT_TYPES}"
        )

        try:
            top3, messages, all_messages_str = get_predictions(graph, query_str, a_str, r_str, d_str, debug=debug)
        except Exception as e:
            logging.error(f"Exception at row {idx}: {e}", exc_info=True)
            top3, messages, all_messages_str = ["", "", ""], [], ""
        
        preds1.append(top3[0])
        preds2.append(top3[1])
        preds3.append(top3[2])
        raw_outputs.append(messages[-1].content if messages else "")
        all_messages_list.append(all_messages_str)
        time.sleep(sleep_time)

    rouge = evaluate.load("rouge")
    scores1 = rouge.compute(
        predictions=preds1,
        references=refs,
        use_stemmer=True,
        rouge_types=["rouge1"],
        use_aggregator=False
    )["rouge1"] # type: ignore
    scores2 = rouge.compute(
        predictions=preds2,
        references=refs,
        use_stemmer=True,
        rouge_types=["rouge1"],
        use_aggregator=False
    )["rouge1"] # type: ignore
    scores3 = rouge.compute(
        predictions=preds3,
        references=refs,
        use_stemmer=True,
        rouge_types=["rouge1"],
        use_aggregator=False
    )["rouge1"] # type: ignore

    best_scores = [max(a, b, c) for a, b, c in zip(scores1, scores2, scores3)]

    df["pred1"], df["pred2"], df["pred3"] = preds1, preds2, preds3
    df["rouge1_1"], df["rouge1_2"], df["rouge1_3"], df["rouge1_best"] = (
        scores1, scores2, scores3, best_scores
    )
    df["raw_output"] = raw_outputs
    df["all_agent_messages"] = all_messages_list

    print(f"Avg ROUGE-1 @rank1:    {sum(scores1)/len(scores1):.4f}")
    print(f"Avg ROUGE-1 @rank2:    {sum(scores2)/len(scores2):.4f}")
    print(f"Avg ROUGE-1 @rank3:    {sum(scores3)/len(scores3):.4f}")
    print(f"Avg ROUGE-1 best-of-3: {sum(best_scores)/len(best_scores):.4f}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate agent predictions.')
    parser.add_argument('--graph_path', '-g', type=str, required=True, help='Path to graph pickle file')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str, required=False, help='Optional: output results file')
    parser.add_argument('--sleep_time', '-s', type=float, default=1.0, help='Sleep time between predictions (in seconds)')
    parser.add_argument('--debug', action="store_true", help="Print agent outputs for each row")
    args = parser.parse_args()

    evaluate_agent(
        graph_pkl=args.graph_path,
        input_csv=args.input,
        output_csv=args.output,
        sleep_time=args.sleep_time,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
