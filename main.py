import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas import ClarificationAnswer, ClarificationAnswers, DecisionRequest
from pipeline import run_mvp


def _ask_answers_interactively(questions) -> ClarificationAnswers:
    print("\n=== Clarifying Questions ===")
    answers = []
    for q in questions:
        print(f"\n[{q.id}] ({q.category}) {q.question}")
        if q.options:
            print("Options:", " | ".join(q.options))
        answer = input("Your answer: ").strip()
        if answer:
            answers.append(ClarificationAnswer(question_id=q.id, answer=answer))
    return ClarificationAnswers(answers=answers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--narrative", type=str, default="")
    parser.add_argument("--use_questioner", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=2)
    args = parser.parse_args()

    title = args.title.strip() or input("Decision title: ").strip()
    narrative = args.narrative.strip() or input("Decision narrative: ").strip()
    req = DecisionRequest(title=title, narrative=narrative)

    all_answers = ClarificationAnswers()
    current_iteration = 0
    previous_questions = []
    iteration_history = []

    while True:
        out = run_mvp(
            req=req,
            use_questioner=args.use_questioner,
            clarification_answers=all_answers,
            current_iteration=current_iteration,
            max_iterations=args.max_iterations,
            previous_questions=previous_questions,
            iteration_history=iteration_history,
        )

        previous_questions = list(out.meta.clarifying_questions)
        iteration_history = list(out.meta.iteration_history)

        if not out.meta.pending_clarification or not out.meta.clarifying_questions:
            print(json.dumps(out.model_dump(), ensure_ascii=False, indent=2))
            break

        new_answers = _ask_answers_interactively(out.meta.clarifying_questions)
        if new_answers.answers:
            all_answers.answers.extend(new_answers.answers)
        current_iteration = out.meta.current_iteration


if __name__ == "__main__":
    main()