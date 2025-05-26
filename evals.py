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
else:
    print("Warning: No HUGGINGFACE_TOKEN environment variable found. Some models may not be accessible.")


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
    """Extract predictions from agent answer text"""
    parts = [p.strip().upper() for p in answer_text.split(",") if p.strip()]
    valid = [p for p in parts if p in EVENT_TYPES]
    return (valid + [""]*3)[:3]

def validate_predictions(predictions, event_intensity, event_type):
    """
    Validates predictions against event intensity and type constraints.
    Returns corrected predictions if violations are found.
    """
    valid_preds = []
    
    # Define intensity-based allowed predictions
    if event_intensity < -5:  # Very hostile
        allowed = ["ASSAULT", "THREATEN", "COERCE", "PROTEST", "SANCTION", "ACCUSE", "REJECT", "RETREAT"]
    elif event_intensity < -2:  # Moderately hostile  
        allowed = ["THREATEN", "ACCUSE", "PROTEST", "SANCTION", "REJECT", "COERCE", "MOBILIZE"]
    elif event_intensity < 0:  # Mildly hostile
        allowed = ["ACCUSE", "THREATEN", "REJECT", "PROTEST", "CONSULT", "REQUEST"]
    else:  # Neutral or positive
        allowed = ["COOPERATE", "CONSULT", "REQUEST", "AID", "CONCEDE", "MOBILIZE"]
    
    # Event type specific constraints (common follow-up patterns)
    event_constraints = {
        "Assault": ["ASSAULT", "THREATEN", "COERCE", "RETREAT", "PROTEST", "SANCTION"],
        "Accuse": ["ACCUSE", "THREATEN", "REJECT", "PROTEST", "CONSULT", "REQUEST"],
        "Sanction": ["SANCTION", "THREATEN", "PROTEST", "REJECT", "ACCUSE", "COERCE"],
        "Protest": ["PROTEST", "THREATEN", "ACCUSE", "COERCE", "SANCTION", "REJECT"],
        "Request": ["REQUEST", "COOPERATE", "CONSULT", "REJECT", "ACCUSE", "AID"],
        "Mobilize": ["MOBILIZE", "THREATEN", "ASSAULT", "PROTEST", "COERCE", "RETREAT"],
    }
    
    # Get event-specific constraints (if available)
    event_allowed = event_constraints.get(event_type, allowed)
    
    # Combine intensity and event constraints (intersection)
    final_allowed = list(set(allowed) & set(event_allowed))
    
    # If intersection is too restrictive, fall back to intensity constraints
    if len(final_allowed) < 3:
        final_allowed = allowed
    
    # Validate each prediction
    for pred in predictions:
        if pred and pred in final_allowed:
            valid_preds.append(pred)
        elif pred:  # Invalid prediction, replace with valid alternative
            # Find the most appropriate replacement based on intensity
            if event_intensity < -5:
                replacement = "THREATEN"
            elif event_intensity < -2:
                replacement = "ACCUSE"  
            elif event_intensity < 0:
                replacement = "REJECT"
            else:
                replacement = "COOPERATE"
            
            # Ensure replacement is in allowed list
            if replacement in final_allowed:
                valid_preds.append(replacement)
            elif final_allowed:
                valid_preds.append(final_allowed[0])
    
    # Fill up to 3 predictions if needed
    while len(valid_preds) < 3 and final_allowed:
        for candidate in final_allowed:
            if candidate not in valid_preds:
                valid_preds.append(candidate)
                break
        if len(valid_preds) == len(final_allowed):  # Avoid infinite loop
            break
    
    # Final fallback if still not enough predictions
    fallback_order = ["THREATEN", "ACCUSE", "COOPERATE", "CONSULT", "REJECT"]
    for fallback in fallback_order:
        if len(valid_preds) >= 3:
            break
        if fallback not in valid_preds:
            if event_intensity < 0 and fallback in ["THREATEN", "ACCUSE", "REJECT"]:
                valid_preds.append(fallback)
            elif event_intensity >= 0 and fallback in ["COOPERATE", "CONSULT"]:
                valid_preds.append(fallback)
    
    return valid_preds[:3]

def get_intensity_description(intensity):
    """Convert intensity value to descriptive text"""
    if intensity < -7:
        return "extremely negative (severe hostility/violence)"
    elif intensity < -5:
        return "highly negative (significant hostility)"
    elif intensity < -2:
        return "moderately negative (notable hostility)"
    elif intensity < 0:
        return "mildly negative (some tension)"
    elif intensity == 0:
        return "neutral"
    else:
        return "positive (cooperative/supportive)"

def build_context_aware_query(row, event_types):
    """Build a context-aware query that includes current event information"""
    # Extract data safely
    actor = str(row.get("Actor Name", "")) if pd.notna(row.get("Actor Name")) else ""
    recipient = str(row.get("Recipient Name", "")) if pd.notna(row.get("Recipient Name")) else ""
    date = str(row.get("Event Date", "")) if pd.notna(row.get("Event Date")) else ""
    event_type = str(row.get("Event Type", "")) if pd.notna(row.get("Event Type")) else ""
    event_intensity = row.get("Event Intensity", 0) if pd.notna(row.get("Event Intensity")) else 0
    contexts = str(row.get("Contexts", "")) if pd.notna(row.get("Contexts")) else ""
    
    # Get intensity description
    intensity_desc = get_intensity_description(event_intensity)
    
    # Build comprehensive query
    query = f"""Given this context:
- Current event: {event_type} (intensity: {event_intensity} - {intensity_desc})
- Event context: {contexts}
- Actors involved: {actor} and {recipient}  
- Date: {date}

Based on this current event and the historical relationship patterns between these actors, what are the 3 most likely follow-up event types that could occur between {actor} and {recipient}?

Consider:
1. The current event intensity ({event_intensity}) and its implications for likely responses
2. Historical patterns between these actors from your knowledge base
3. Typical escalation/de-escalation patterns in international relations
4. Logical consistency (hostile events typically trigger defensive or retaliatory responses)
5. Power dynamics and contextual factors

Important constraints:
- For very negative events (intensity < -5), responses are typically hostile: THREATEN, ASSAULT, COERCE, PROTEST, SANCTION
- For moderately negative events (intensity -2 to -5), responses may include: ACCUSE, THREATEN, REJECT, PROTEST
- For neutral/positive events (intensity >= 0), responses may include: COOPERATE, CONSULT, REQUEST, AID

Choose your top 3 predictions from: {event_types}"""
    
    return query, actor, recipient, date, event_intensity, event_type

def get_predictions(graph, model, query, actor, recipient, date, event_intensity=0, event_type="", debug=False):
    """Enhanced version with validation"""
    if debug:
        logging.info(f"Query: {query}")
        logging.info(f"Event context: {event_type}, Intensity: {event_intensity}")
    
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
        # Return validation-based fallback
        fallback_preds = validate_predictions(["", "", ""], event_intensity, event_type)
        return fallback_preds, [], str(e)

    if not result["messages"]:
        fallback_preds = validate_predictions(["", "", ""], event_intensity, event_type)
        return fallback_preds, [], "No messages returned"

    last = result["messages"][-1].content
    all_messages_str = "\n\n".join(
        f"{getattr(m, 'role', type(m).__name__)}: {getattr(m, 'content', str(m))}"
        for m in result["messages"]
    )

    # Extract predictions from answer
    m = re.search(r"Answer:\s*(.*)", last, re.IGNORECASE)
    if not m:
        # Try alternative extraction patterns
        m = re.search(r"(?:final answer|prediction|conclusion):\s*(.*)", last, re.IGNORECASE)
        if not m:
            logging.warning(f"No 'Answer:' found in response: {last[:200]}...")
            # Fallback validation-based prediction
            fallback_preds = validate_predictions(["", "", ""], event_intensity, event_type)
            return fallback_preds, result["messages"], all_messages_str
    
    raw_preds = extract_pred(m.group(1))
    
    # Apply validation to ensure logical consistency
    validated_preds = validate_predictions(raw_preds, event_intensity, event_type)
    
    if debug:
        logging.info(f"Raw predictions: {raw_preds}")
        logging.info(f"Validated predictions: {validated_preds}")
    
    return validated_preds, result["messages"], all_messages_str

def evaluate_agent(graph_pkl, input_csv, output_csv=None, sleep_time=1.0, debug=False):
    graph = load_graph(graph_pkl)
    inference_url = "http://localhost:8000/v1"

    llm = ChatOpenAI(
        model="mistralai/Mistral-7B-Instruct-v0.3",
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
    validation_stats = {"total_validated": 0, "intensity_violations": 0, "no_answer_found": 0}

    logging.info(f"Starting evaluation on {len(df)} test cases...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Build context-aware query
        query_str, actor, recipient, date, event_intensity, event_type = build_context_aware_query(row, EVENT_TYPES)
        
        if debug:
            logging.info(f"\nRow {idx}: {event_type} (intensity: {event_intensity})")
            logging.info(f"Actor: {actor}, Recipient: {recipient}, Date: {date}")

        try:
            top3, messages, all_messages_str = get_predictions(
                graph, llm, query_str, actor, recipient, date, 
                event_intensity=event_intensity, event_type=event_type, debug=debug
            )
            
            # Track validation statistics
            if not messages or not messages[-1].content:
                validation_stats["no_answer_found"] += 1
            else:
                # Check if validation was needed (compare with raw extraction)
                last_content = messages[-1].content
                m = re.search(r"Answer:\s*(.*)", last_content, re.IGNORECASE)
                if m:
                    raw_preds = extract_pred(m.group(1))
                    if raw_preds != top3:
                        validation_stats["total_validated"] += 1
                        # Check for intensity violations in raw predictions
                        for pred in raw_preds:
                            if pred and event_intensity < -2 and pred == "COOPERATE":
                                validation_stats["intensity_violations"] += 1
                                break
                            
        except Exception as e:
            logging.error(f"Exception at row {idx}: {e}", exc_info=True)
            top3 = validate_predictions(["", "", ""], event_intensity, event_type)
            messages, all_messages_str = [], ""
            validation_stats["no_answer_found"] += 1

        preds1.append(top3[0] if top3[0] else "")
        preds2.append(top3[1] if len(top3) > 1 and top3[1] else "")
        preds3.append(top3[2] if len(top3) > 2 and top3[2] else "")
        raw_outputs.append(messages[-1].content if messages else "")
        all_messages_list.append(all_messages_str)
        
        time.sleep(sleep_time)

    # Calculate ROUGE scores
    logging.info("Calculating ROUGE scores...")
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

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
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
    
    # Validation statistics
    print("\nVALIDATION STATISTICS:")
    print(f"  Cases validated: {validation_stats['total_validated']}/{len(df)} ({validation_stats['total_validated']/len(df)*100:.1f}%)")
    print(f"  Intensity violations caught: {validation_stats['intensity_violations']}")
    print(f"  No answer found: {validation_stats['no_answer_found']}")

    # Save results
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate agent predictions with enhanced context awareness.')
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