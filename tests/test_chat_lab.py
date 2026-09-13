import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from text_factors.chat_lab import ContextChatSession, make_chat_demo_model
from text_factors.cli import main
from text_factors.dialogue import GroundedDialogue


class ChatLabTests(unittest.TestCase):
    def test_names_are_learned_and_survive_new_arrangement_and_reload(self) -> None:
        model = make_chat_demo_model()
        session = ContextChatSession(model)
        before = model.memory.stats()
        self.assertEqual(session.handle("что значит марс").kind, "clarification")
        session.show(("ab",), reference_id="a")
        session.handle("назови 1 марс")
        session.show(("cd",), reference_id="b")
        session.handle("назови 1 нептун")
        reply = session.show(("cd", "ab"), reference_id="combined")
        self.assertEqual(reply.kind, "description")
        self.assertIn("марс", reply.text)
        self.assertIn("нептун", reply.text)
        restored = GroundedDialogue.from_dict(session.dialogue.to_dict())
        self.assertEqual(restored.describe("combined"), reply)
        self.assertEqual(model.memory.stats(), before)
        self.assertEqual(session.dialogue.stats()["events"], 2)

    def test_saved_dialogue_rejects_another_encoder(self) -> None:
        session = ContextChatSession(make_chat_demo_model(7))
        with self.assertRaisesRegex(ValueError, "another model"):
            ContextChatSession(make_chat_demo_model(17), session.dialogue)

    def test_chat_cli_saves_and_resumes_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = str(Path(directory) / "chat.json")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        "chat",
                        "--demo",
                        "--state",
                        state,
                        "--message",
                        "покажи ab",
                        "--message",
                        "назови 1 марс",
                    ]
                )
                resumed = main(
                    [
                        "chat",
                        "--demo",
                        "--state",
                        state,
                        "--message",
                        "покажи ab",
                        "--message",
                        "что значит марс",
                    ]
                )
            self.assertEqual((result, resumed), (0, 0))
            self.assertIn("Вижу: марс.", output.getvalue())
            self.assertEqual(GroundedDialogue.load(state).stats()["events"], 1)

    def test_recognize_cli_exposes_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            make_chat_demo_model().save(model_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        "recognize",
                        "--model",
                        str(model_path),
                        "--text",
                        "ab cd",
                        "--max-views",
                        "1",
                    ]
                )
            data = json.loads(output.getvalue())
            self.assertEqual(result, 1)
            self.assertFalse(data["complete"])
            self.assertEqual(data["stop_reason"], "view_limit")


if __name__ == "__main__":
    unittest.main()
