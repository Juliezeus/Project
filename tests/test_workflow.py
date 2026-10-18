import json
import unittest
from pathlib import Path

from hr_rag.corpus import evidence_ids, load_corpus
from hr_rag.retrieval import rrf
from hr_rag.workflow import build_workflow, local_context
from hr_rag.workflow import Context, Answer, Claim, Review
from langchain_core.runnables import RunnableLambda


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus()
        cls.app = build_workflow(cls.docs)

    def test_corpus(self):
        self.assertEqual(len(self.docs), 22)
        for doc in self.docs.values():
            self.assertEqual(doc.metadata["version"], "1.1")

    def test_required_evidence(self):
        cases = {
            "annual": {"LEA01", "BU02"},
            "learning": {"BEN02", "BU03"},
            "wellbeing": {"BEN01", "BU03"},
            "bonus": {"REW01", "REW02", "BU04", "PAY02"},
            "parental": {"LEA03"},
        }
        for topic, required in cases.items():
            with self.subTest(topic=topic):
                self.assertTrue(required <= set(evidence_ids([topic], ["OPS02"])))

    def test_transfer_includes_history_rule(self):
        result = self.app.invoke({"question": "Moving from corporate to technology, what is my learning budget?"})
        self.assertTrue({"BU05", "BU03", "BEN02"} <= set(result["evidence"]))

    def test_multiple_benefits(self):
        result = self.app.invoke({"question": "0.6 FTE technology employee: wellbeing and learning budgets?"})
        self.assertTrue({"BEN01", "BEN02", "BU03"} <= set(result["evidence"]))
        self.assertEqual(result["context"]["fte"], 0.6)
        self.assertEqual(result["status"], "evidence_only")
        self.assertFalse(result["answer"]["claims"])

    def test_no_fte_guess(self):
        ctx = local_context("I work three days per week; what is my wellbeing budget?", {})
        self.assertIsNone(ctx["fte"])

    def test_rank_fusion_rewards_agreement(self):
        self.assertEqual(rrf(["LEA01", "BU02"], ["BU02", "BEN01"])[0], "BU02")

    def test_bonus_clarification_cannot_skip_policy_retrieval(self):
        called = []

        class Model:
            def with_structured_output(self, schema, **kwargs):
                called.append(schema.__name__)
                values = {
                    Context: Context(topics=["bonus"], missing=["special_bonus_eligible"]),
                    Answer: Answer(claims=[Claim(text="Bonus targets depend on grade and BU; awards are not guaranteed.", citations=["BU04", "REW01"])], clarification="What are your home BU and job grade?"),
                    Review: Review(supported=True, applicable=True, complete=True, reason="Supported general rules and appropriate clarification."),
                }
                return RunnableLambda(lambda _: values[schema])

        result = build_workflow(self.docs, model=Model()).invoke({"question": "What is my bonus entitlement?"})
        self.assertTrue({"REW01", "REW02", "REW03", "BU04"} <= set(result["evidence"]))
        self.assertEqual(result["context"]["missing"], [])
        self.assertTrue(result["answer"]["claims"])
        self.assertEqual(called, ["Context", "Answer", "Review"])

    def test_draft_evaluation_retrieval_coverage(self):
        cases = json.loads((Path(__file__).resolve().parents[1] / "evaluation/cases.json").read_text())
        for case in cases:
            with self.subTest(case=case["id"]):
                result = self.app.invoke({"question": case["question"]})
                self.assertTrue(set(case["required_sections"]) <= set(result["evidence"]))

    def test_unknown_question_remains_evidence_only(self):
        result = self.app.invoke({"question": "Will my manager approve a mortgage?"})
        self.assertEqual(result["status"], "evidence_only")
        self.assertEqual(result["context"]["topics"], [])


if __name__ == "__main__":
    unittest.main()
