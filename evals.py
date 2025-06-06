from create_agent import run_agent
from langchain_openai import ChatOpenAI
from utils.load_graph import load_graph
from huggingface_hub import login

from collections import Counter
import argparse
import pandas as pd
import regex as re
import time
from tqdm import tqdm
import evaluate
import logging
import sys
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACEHUB_API_TOKEN:
    login(token=HUGGINGFACEHUB_API_TOKEN)

EVENT_TYPES = (
    "ACCUSE", "ASSAULT", "AID", "REQUEST", "PROTEST", "COERCE", "THREATEN",
    "RETREAT", "MOBILIZE", "SANCTION", "CONCEDE", "COOPERATE",
    "CONSULT", "REJECT"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def extract_pred(answer_text):
    """Extract predictions from agent answer text"""
    parts = [p.strip().upper() for p in answer_text.split(",") if p.strip()]
    valid = [p for p in parts if p in EVENT_TYPES]
    return (valid + [""]*3)[:3]

def build_query(row, event_types):
    """Build a context-aware query that includes current event information"""
    actor = str(row.get("Actor Name", "")) if pd.notna(row.get("Actor Name")) else ""
    recipient = str(row.get("Recipient Name", "")) if pd.notna(row.get("Recipient Name")) else ""
    date = str(row.get("Event Date", "")) if pd.notna(row.get("Event Date")) else ""
    event_type = str(row.get("Event Type", "")) if pd.notna(row.get("Event Type")) else ""
    event_intensity = row.get("Event Intensity", 0) if pd.notna(row.get("Event Intensity")) else 0
    contexts = str(row.get("Contexts", "")) if pd.notna(row.get("Contexts")) else ""
    
    # Get intensity description
    if event_intensity < -7:
        intensity_desc = "extremely negative (severe hostility/violence)"
    elif event_intensity < -5:
        intensity_desc = "highly negative (significant hostility)"
    elif event_intensity < -2:
        intensity_desc = "moderately negative (notable hostility)"
    elif event_intensity < 0:
        intensity_desc = "mildly negative (some tension)"
    elif event_intensity == 0:
        intensity_desc = "neutral"
    else:
        intensity_desc = "positive (cooperative/supportive)"
    
    # Build comprehensive query that matches your system prompt expectations
    query = f"""Given this context:
- Current event: {event_type} (intensity: {event_intensity} - {intensity_desc})
- Event context: {contexts}
- Actors involved: {actor} and {recipient}  
- Date: {date}

Based on this current event and the historical relationship patterns between these actors, what are the 3 most likely follow-up event types that could occur between {actor} and {recipient}?

Consider the current event intensity ({event_intensity}) and apply the intensity-based prediction rules from your guidelines.

Choose your top 3 predictions from: {event_types}"""
    
    return query, actor, recipient, date, event_intensity, event_type

def get_agent_predictions(graph, model, query, actor, recipient, date, debug=False):
    """Get agent predictions WITHOUT any validation or correction"""
    if debug:
        logging.info(f"Query: {query}")
    
    try:
        result = run_agent(
            graph=graph,
            query=query,
            model=model,
            actor=actor,
            recipient=recipient,
            date=date
        )
    except Exception as e:
        logging.error(f"Error running agent: {e}", exc_info=True)
        return ["", "", ""], [], str(e)

    if not result["messages"]:
        return ["", "", ""], [], "No messages returned"

    last = result["messages"][-1].content
    all_messages_str = "\n\n".join(
        f"{getattr(m, 'role', type(m).__name__)}: {getattr(m, 'content', str(m))}"
        for m in result["messages"]
    )

    # Extract predictions from answer - NO VALIDATION
    m = re.search(r"Answer:\s*(.*)", last, re.IGNORECASE)
    if not m:
        # Try alternative extraction patterns
        m = re.search(r"(?:final answer|prediction|conclusion):\s*(.*)", last, re.IGNORECASE)
        if not m:
            logging.warning(f"No 'Answer:' found in response: {last[:200]}...")
            return ["", "", ""], result["messages"], all_messages_str
    
    # Return RAW predictions - let the agent succeed or fail on its own merit
    raw_preds = extract_pred(m.group(1))
    
    if debug:
        logging.info(f"Agent predictions: {raw_preds}")
    
    return raw_preds, result["messages"], all_messages_str

def analyze_agent_reasoning(df_results):
    """Analyze agent's logical reasoning patterns without correcting them"""
    
    analysis = {
        'total_predictions': 0,
        'logical_violations': 0,
        'intensity_violations': 0,
        'empty_predictions': 0,
        'valid_format': 0
    }
    
    violation_examples = []
    
    for idx, row in df_results.iterrows():
        event_type = row.get('Event Type', '')
        event_intensity = row.get('Event Intensity', 0)
        pred1 = row.get('pred1', '')
        
        analysis['total_predictions'] += 1
        
        if not pred1:
            analysis['empty_predictions'] += 1
            continue
            
        if pred1 in EVENT_TYPES:
            analysis['valid_format'] += 1
        
        # Check for logical violations (for analysis only, NOT correction)
        violation = None
        if event_intensity < -2 and pred1 == "COOPERATE":
            violation = "cooperative_after_hostility"
            analysis['intensity_violations'] += 1
        elif event_intensity > 2 and pred1 in ["ASSAULT", "THREATEN"]:
            violation = "hostility_after_cooperation"
            analysis['intensity_violations'] += 1
        elif event_type == "Assault" and pred1 == "COOPERATE":
            violation = "cooperative_after_assault"
            analysis['logical_violations'] += 1
        
        if violation:
            violation_examples.append({
                'row': idx,
                'event_type': event_type,
                'intensity': event_intensity,
                'prediction': pred1,
                'violation_type': violation
            })
    
    return analysis, violation_examples

def evaluate_agent(graph_pkl, input_csv, output_csv=None, sleep_time=1.0, debug=False):
    """Clean evaluation of agent performance without any prediction validation"""
    graph = load_graph(graph_pkl)
    inference_url = "http://localhost:8000/v1"

    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        openai_api_key="EMPTY", # type: ignore
        openai_api_base=inference_url, # type: ignore
        temperature=0,
    )

    # Load test data
    df = pd.read_csv(input_csv)
    refs = df["Event Type"].astype(str).str.upper().tolist()

    # Initialize result containers
    preds1, preds2, preds3 = [], [], []
    raw_outputs = []
    all_messages_list = []
    failed_queries = 0

    logging.info(f"Starting clean evaluation on {len(df)} test cases...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Build context-aware query
        query_str, actor, recipient, date, event_intensity, event_type = build_query(row, EVENT_TYPES)
        
        if debug:
            logging.info(f"\nRow {idx}: {event_type} (intensity: {event_intensity})")
            logging.info(f"Actor: {actor}, Recipient: {recipient}, Date: {date}")

        try:
            # Get agent predictions - NO VALIDATION OR CORRECTION
            top3, messages, all_messages_str = get_agent_predictions(
                graph, llm, query_str, actor, recipient, date, debug=debug
            )
            
            if not top3[0]:  # Empty prediction
                failed_queries += 1
                            
        except Exception as e:
            logging.error(f"Exception at row {idx}: {e}", exc_info=True)
            top3 = ["", "", ""]
            messages, all_messages_str = [], ""
            failed_queries += 1

        # Store RAW predictions
        preds1.append(top3[0] if len(top3) > 0 and top3[0] else "")
        preds2.append(top3[1] if len(top3) > 1 and top3[1] else "")
        preds3.append(top3[2] if len(top3) > 2 and top3[2] else "")
        raw_outputs.append(messages[-1].content if messages else "")
        all_messages_list.append(all_messages_str)
        
        time.sleep(sleep_time)

    # Calculate ROUGE scores on RAW agent predictions
    logging.info("Calculating ROUGE scores on raw agent predictions...")
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

    # Add results to dataframe
    df["pred1"], df["pred2"], df["pred3"] = preds1, preds2, preds3
    df["rouge1_1"], df["rouge1_2"], df["rouge1_3"], df["rouge1_best"] = (
        scores1, scores2, scores3, best_scores
    )
    df["raw_output"] = raw_outputs
    df["all_agent_messages"] = all_messages_list

    # Analyze agent reasoning patterns
    analysis, violation_examples = analyze_agent_reasoning(df)

    # Print results
    print("\n" + "="*60)
    print("PURE AGENT EVALUATION RESULTS")
    print("="*60)
    print(f"Avg ROUGE-1 @rank1:    {sum(scores1)/len(scores1):.4f}")
    print(f"Avg ROUGE-1 @rank2:    {sum(scores2)/len(scores2):.4f}")
    print(f"Avg ROUGE-1 @rank3:    {sum(scores3)/len(scores3):.4f}")
    print(f"Avg ROUGE-1 best-of-3: {sum(best_scores)/len(best_scores):.4f}")
    
    # Prediction distribution analysis
    print("\nPREDICTION DISTRIBUTION:")
    pred1_counts = Counter([p for p in preds1 if p])
    for pred, count in pred1_counts.most_common():
        percentage = (count / len(preds1)) * 100
        print(f"  {pred}: {count}/{len(preds1)} ({percentage:.1f}%)")
    
    # Agent reasoning analysis
    print("\nAGENT REASONING ANALYSIS:")
    print(f"  Total predictions: {analysis['total_predictions']}")
    print(f"  Failed queries: {failed_queries} ({failed_queries/len(df)*100:.1f}%)")
    print(f"  Valid format: {analysis['valid_format']}/{analysis['total_predictions']} ({analysis['valid_format']/analysis['total_predictions']*100:.1f}%)")
    print(f"  Intensity violations: {analysis['intensity_violations']} ({analysis['intensity_violations']/analysis['total_predictions']*100:.1f}%)")
    print(f"  Logic violations: {analysis['logical_violations']} ({analysis['logical_violations']/analysis['total_predictions']*100:.1f}%)")
    
    if violation_examples[:3]:  # Show first 3 examples
        print("\nExample violations:")
        for ex in violation_examples[:3]:
            print(f"  Row {ex['row']}: {ex['event_type']} (intensity {ex['intensity']}) → {ex['prediction']} [{ex['violation_type']}]")

    # Save results
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved results to {output_csv}")

    return {
        'rouge_scores': {
            'rank1': sum(scores1)/len(scores1),
            'rank2': sum(scores2)/len(scores2), 
            'rank3': sum(scores3)/len(scores3),
            'best': sum(best_scores)/len(best_scores)
        },
        'analysis': analysis,
        'violations': violation_examples
    }

def main():
    parser = argparse.ArgumentParser(description='Clean agent evaluation without validation interference.')
    parser.add_argument('--graph_path', '-g', type=str, required=True, help='Path to graph pickle file')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str, required=False, help='Optional: output results file')
    parser.add_argument('--sleep_time', '-s', type=float, default=1.0, help='Sleep time between predictions (in seconds)')
    parser.add_argument('--debug', action="store_true", help="Print agent outputs for each row")
    args = parser.parse_args()

    results = evaluate_agent(
        graph_pkl=args.graph_path,
        input_csv=args.input,
        output_csv=args.output,
        sleep_time=args.sleep_time,
        debug=args.debug
    )
    
    return results

if __name__ == "__main__":
    main()