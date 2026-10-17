"""P16b: per-model context budgets in tokens, and more excerpts when the budget is bigger (all additive)."""

from django.test import SimpleTestCase, override_settings

from chat.views import context_budget_chars, rag_top_n, uses_token_budget


class TokenBudgetTests(SimpleTestCase):
    @override_settings(CHAT_CONTEXT_MAX_CHARS=24000, CHAT_CONTEXT_MAX_CHARS_OLLAMA=6000,
                       CHAT_CONTEXT_MAX_TOKENS_DGX=0, CHAT_CONTEXT_MAX_TOKENS_OLLAMA=0)
    def test_unset_tokens_keep_the_char_budgets(self):
        self.assertEqual(context_budget_chars("dgx-qwen38-27b-fp8"), 24000)
        self.assertEqual(context_budget_chars("ollama-qwen3-8b"), 6000)

    @override_settings(CHAT_CONTEXT_MAX_CHARS=24000, CHAT_CONTEXT_MAX_TOKENS_DGX=16384, CHAT_CONTEXT_MAX_TOKENS_OLLAMA=0,
                       CHAT_ANSWER_RESERVE_TOKENS=4096, CHAT_CHARS_PER_TOKEN=3.0)
    def test_tokens_set_the_prompt_budget(self):
        self.assertEqual(context_budget_chars("dgx-qwen38-27b-fp8"), (16384 - 4096) * 3)
        self.assertEqual(context_budget_chars("dgx-gpt-oss-20b"), (16384 - 4096) * 3)
        self.assertEqual(context_budget_chars("ollama-qwen3-8b"), 6000)    # its own setting is not set

    @override_settings(CHAT_CONTEXT_MAX_TOKENS_DGX=6000, CHAT_ANSWER_RESERVE_TOKENS=4096, CHAT_CHARS_PER_TOKEN=3.0)
    def test_the_reserve_never_takes_more_than_half(self):
        self.assertEqual(context_budget_chars("dgx-qwen38-27b-fp8"), 3000 * 3)

    @override_settings(RAG_CHAT_TOP_N=5, RAG_V2_CANDIDATES=20, RAG_CHAT_CHARS_PER_CHUNK=1500)
    def test_more_chunks_for_a_bigger_excerpt_budget(self):
        self.assertEqual(rag_top_n(1500), 5)          # the 8B today
        self.assertEqual(rag_top_n(6000), 5)          # the 27B today
        self.assertEqual(rag_top_n(9216), 6)
        self.assertEqual(rag_top_n(27648), 18)
        self.assertEqual(rag_top_n(100000), 20)       # never more than the reranker's candidates

    @override_settings(RAG_CHAT_TOP_N=5, CHAT_CONTEXT_MAX_TOKENS_DGX=16384, CHAT_CONTEXT_MAX_TOKENS_OLLAMA=0)
    def test_only_a_token_budget_grows_the_chunks(self):
        self.assertTrue(uses_token_budget("dgx-qwen38-27b-fp8"))
        self.assertFalse(uses_token_budget("ollama-qwen3-8b"))
        self.assertEqual(rag_top_n(12000, grow=False), 5)    # a library chat today keeps its 5 chunks
