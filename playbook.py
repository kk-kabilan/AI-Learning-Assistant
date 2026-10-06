import json
import os


PLAYBOOK_FILE = "playbook.json"


# =========================================================
# LOAD PLAYBOOK
# =========================================================

def load_playbook():

    if not os.path.exists(PLAYBOOK_FILE):
        return []

    with open(PLAYBOOK_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# SAVE PLAYBOOK
# =========================================================

def save_playbook(playbook):

    with open(PLAYBOOK_FILE, "w", encoding="utf-8") as file:

        json.dump(
            playbook,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# ADD STRUCTURED ENTRY
# =========================================================

def add_to_playbook(query, answer, context):

    playbook = load_playbook()

    entry = {
        "id": f"pb_{len(playbook) + 1:04d}",
        "topic": "General",
        "question": query,
        "knowledge": answer,
        "source": context,
        "verified": True,
        "usage_count": 0,
        "helpful_count": 0,
        "harmful_count": 0
    }

    playbook.append(entry)

    save_playbook(playbook)

    print("Playbook updated.")


# =========================================================
# GET TEXT USED FOR SEMANTIC SEARCH
# =========================================================

def get_search_text(entry):

    # New structured entries
    if "knowledge" in entry:
        return (
            str(entry.get("question", "")) + " " +
            str(entry.get("knowledge", "")) + " " +
            str(entry.get("topic", ""))
        )

    # Compatibility with old Playbook entries
    return (
        str(entry.get("question", "")) + " " +
        str(entry.get("answer", ""))
    )


# =========================================================
# SEMANTIC PLAYBOOK SEARCH
# BACKWARD-COMPATIBLE VERSION
# =========================================================

def search_playbook(query, model, threshold=0.60):

    results = search_playbook_top_k(
        query=query,
        model=model,
        k=1,
        threshold=threshold
    )

    if not results:
        return None

    return results[0]["entry"]


# =========================================================
# TOP-K SEMANTIC PLAYBOOK SEARCH
# =========================================================

def search_playbook_top_k(
    query,
    model,
    k=3,
    threshold=0.60
):

    playbook = load_playbook()

    if not playbook:
        return []

    # Create embedding for the new question
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scored_entries = []

    for entry in playbook:

        search_text = get_search_text(entry)

        stored_embedding = model.encode(
            [search_text],
            normalize_embeddings=True
        )[0]

        # Cosine similarity
        score = float(
            query_embedding @ stored_embedding
        )

        scored_entries.append({
            "entry": entry,
            "score": score
        })

    # Highest similarity first
    scored_entries.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Remove results below threshold
    filtered_results = [
        result
        for result in scored_entries
        if result["score"] >= threshold
    ]

    # Return only top-k
    top_results = filtered_results[:k]

    print("Playbook candidates:", len(scored_entries))

    if top_results:
        print("Top Playbook similarities:")

        for result in top_results:

            print(
                f"  {result['score']:.4f} "
                f"- {result['entry'].get('question', '')}"
            )

    else:

        print("No Playbook entry passed the similarity threshold.")

    return top_results