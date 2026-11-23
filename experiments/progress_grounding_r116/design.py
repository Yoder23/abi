"""Content-seeded counterfactual worlds and analysis. No model or classifier."""
import hashlib
import math
import random
import re

# Claim, actual evidence and prospective foil. Templates are disclosed.
FAMILIES = {
    "started": ("Work on {p} has started.", ("The crew has begun the first shift.", "The crew is working on the first batch."), ("The first shift is on the calendar.", "The crew has a plan for its first batch.")),
    "completed": ("Work on {p} has finished.", ("The entire job is done.", "The last piece of work is complete."), ("The entire job has an assigned team.", "The last piece of work is on the team's list.")),
    "resumed": ("Work on {p} has resumed after a pause.", ("Following the pause, staff are back at work.", "The crew restarted the job after the break."), ("Following the pause, staff have permission to return.", "The crew can restart the job after the break.")),
    "delivered": ("The package for {p} has been delivered.", ("Its package reached the destination.", "The addressee has the package in hand."), ("Its package is ready for the trip.", "The package is packed and awaiting dispatch.")),
    "inspected": ("The site for {p} has been inspected.", ("The inspector finished examining the site.", "The inspection visit took place."), ("The inspector arranged a visit to the site.", "An inspection visit has been booked.")),
    "issued": ("The notice for {p} has been issued.", ("The office released its notice to the public.", "The notice has gone out to subscribers."), ("The office wrote a draft of its notice.", "The notice is ready to go out to subscribers.")),
    "received": ("Payment for {p} has been received.", ("The recipient's account was credited with the payment.", "The recipient has the funds."), ("The payment is due today.", "The recipient expects the funds today.")),
    "on_time": ("Work on {p} actually began at its scheduled time.", ("The scheduled start was noon. The crew began at noon.", "The shift was booked for nine and the crew clocked in at nine."), ("The scheduled start was noon. No start time was observed.", "The shift was booked for nine. The log has no clock-in entry.")),
}
ARMS = ("source", "parent", "abi_plan", "lora", "distillation")
REPEATS = ("initial", "replay")


def cases(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("need exact frozen commit")
    rng = random.Random(int.from_bytes(hashlib.sha256(("R116:" + commit).encode()).digest(), "big"))
    rows = []
    for family, (claim, actual, foil) in FAMILIES.items():
        for index in range(6):
            names = [f"Project {rng.getrandbits(48):012X}" for _ in range(2)]
            order = [0, 1]
            rng.shuffle(order)
            facts = [rng.choice(foil), rng.choice(actual)]
            for world in (0, 1, 2):
                notes = "\n".join(f"{names[n]}: {facts[int(n == world)]}" for n in order) if world < 2 else "(Notes withheld.)"
                for subject in (0, 1):
                    body = ("Use only the notes below. Answer Yes only if they establish the claim for the named project. "
                            "If the evidence is missing, answer No. Do not assume a planned event has happened.\n"
                            f"Notes:\n{notes}\nClaim: {claim.format(p=names[subject])}\nDoes the evidence establish this claim?")
                    rows.append(dict(id=f"{family}-{index}-{world}-{subject}", family=family, pair=index,
                                     world=world, subject=subject, body=body, gold="Yes" if world == subject else "No"))
    if len({r["body"] for r in rows}) != 288:
        raise ValueError("duplicate prompt")
    return rows


def summarize(records, population):
    if len(records) != 288 or [r["id"] for r in records] != [p["id"] for p in population]:
        raise ValueError("missing or reordered observations")
    clusters = {k: [True] * 6 for k in FAMILIES}
    counts = {"evidence_correct": 0, "positive_correct": 0, "foil_false_yes": 0,
              "withheld_incorrect": 0, "ties": 0, "unrestricted_top_correct": 0}
    errors = []
    for r, p in zip(records, population):
        a, b = r["yes"], r["no"]
        if not all(type(x) is float and math.isfinite(x) for x in (a, b)):
            raise ValueError("invalid binary logits")
        prediction = "Yes" if a > b else "No" if b > a else "tie"
        correct = prediction == p["gold"]
        if not correct:
            errors.append(p["id"])
        counts["ties"] += a == b
        counts["unrestricted_top_correct"] += r["top_id"] == r[p["gold"].lower() + "_id"]
        if p["world"] == 2:
            counts["withheld_incorrect"] += not correct
        else:
            counts["evidence_correct"] += correct
            counts["positive_correct"] += correct and p["gold"] == "Yes"
            counts["foil_false_yes"] += prediction == "Yes" and p["gold"] == "No"
            clusters[p["family"]][p["pair"]] &= correct
    per = {k: sum(v) for k, v in clusters.items()}
    total = sum(per.values())
    return {**counts, "errors": errors, "clusters": clusters, "per_family": per, "complete_clusters": total,
            "qualified_diagnostic_only": total >= 44 and min(per.values()) >= 5 and counts["withheld_incorrect"] == 0}


def contrast(candidate, control):
    import numpy as np
    diff = np.array([[int(a)-int(b) for a, b in zip(candidate[k], control[k])] for k in FAMILIES])
    idx = np.random.default_rng(116001).integers(0, 6, size=(10000, 8, 6))
    draws = diff[np.arange(8)[None, :, None], idx].mean(axis=(1, 2))
    return {"mean": float(diff.mean()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}
