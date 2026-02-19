import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import DecisionRequest, ClarificationAnswers, ClarificationAnswer
from pipeline import run_mvp


def _ask_answers_interactively(questions) -> ClarificationAnswers:
    print("\n=== Clarifying Questions ===")
    answers = []
    for q in questions:
        print(f"\n[{q.id}] ({q.category}) {q.question}")
        if q.options:
            print("Options:", " | ".join(q.options))
        ans = input("Your answer: ").strip()
        if ans:
            answers.append(ClarificationAnswer(question_id=q.id, answer=ans))
    return ClarificationAnswers(answers=answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--narrative", type=str, default="")
    parser.add_argument("--use_questioner", action="store_true", help="Enable Questioner clarification stage")
    args = parser.parse_args()

    title = args.title.strip() or input("Decision title: ").strip()
    narrative = args.narrative.strip() or input("Decision narrative: ").strip()

    req = DecisionRequest(title=title, narrative=narrative)

    if not args.use_questioner:
        out = run_mvp(req)
        print(json.dumps(out.model_dump(), ensure_ascii=False, indent=2))
        return

    # Phase 1: get questions
    out1 = run_mvp(req, use_questioner=True)

    if out1.meta.pending_clarification and out1.meta.clarifying_questions:
        clar = _ask_answers_interactively(out1.meta.clarifying_questions)
        out2 = run_mvp(req, use_questioner=True, clarification_answers=clar)
        print(json.dumps(out2.model_dump(), ensure_ascii=False, indent=2))
    else:
        # If no clarification needed, out1 is already a full result (or at least not pending)
        print(json.dumps(out1.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
