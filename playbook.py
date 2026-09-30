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
# ADD ENTRY
# =========================================================

def add_to_playbook(query, answer, context):

    playbook = load_playbook()

    entry = {
        "question": query,
        "answer": answer,
        "context": context
    }

    playbook.append(entry)

    save_playbook(playbook)

    print("Playbook updated.")


# =========================================================
# SEMANTIC PLAYBOOK SEARCH
# =========================================================

def search_playbook(query, model, threshold=0.75):

    playbook = load_playbook()

    if not playbook:
        return None

    # Create embedding for new question
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    best_entry = None
    best_score = -1

    for entry in playbook:

        stored_embedding = model.encode(
            [entry["question"]],
            normalize_embeddings=True
        )[0]

        # Cosine similarity
        score = float(
            query_embedding @ stored_embedding
        )

        if score > best_score:

            best_score = score
            best_entry = entry

    print("Best Playbook similarity:", best_score)

    if best_score >= threshold:

        return best_entry

    return None