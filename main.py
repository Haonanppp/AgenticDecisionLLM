import json
import argparse
from agenticdq.schemas import DecisionRequest
from agenticdq.pipeline import run_mvp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--narrative", type=str, default="")
    args = parser.parse_args()

    title = args.title.strip() or input("Decision title: ").strip()
    narrative = args.narrative.strip() or input("Decision narrative: ").strip()

    req = DecisionRequest(title=title, narrative=narrative)
    out = run_mvp(req)

    print(json.dumps(out.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
