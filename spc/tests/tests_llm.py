"""The LLM analysis of an SPC tool (P12f): the packet, one call to a model, the two checks. No model is called:
the HTTP calls are fakes. The historian is the fake of the evaluator's tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from django.test import SimpleTestCase, override_settings

from spc import llm
from spc.models import SpcEvent
from spc.tests.tests_evaluate import EvaluatorTestCase, at

PACKET = {
    "window": {"start": "2026-10-13T21:00:00Z", "end": "2026-10-14T05:00:00Z"},
    "control_limits": [{"cl": 89.92, "lcl": 79.806, "ucl": 100.03}],
    "now": {"sigma_shift": 0.06},
    "chart": {"fired_by_rule": {"beyond_limits": 2, "no_variation": 1}},
    "capability": {"cpk": 2.0324, "cp": 2.0461},
    "events": {"in_window": 3},
    "distribution": {"mean": 90.135},
}


def ollama_reply(content="1. State of control - in control.", **extra):
    reply = MagicMock()
    reply.json.return_value = {"message": {"content": content}, "done_reason": "stop", "prompt_eval_count": 5098, "eval_count": 412,
                               "load_duration": 5_540_000_000, "prompt_eval_duration": 1_240_000_000, "eval_duration": 4_620_000_000, **extra}
    return reply


class CapacityTests(SimpleTestCase):
    def setUp(self):
        llm._capacity_cache.clear()
        self.addCleanup(llm._capacity_cache.clear)

    def test_the_capacity_is_asked_of_the_model_and_kept_or_taken_from_the_settings(self):
        show = MagicMock(json=MagicMock(return_value={"model_info": {"qwen3.context_length": 40960}}))
        with patch("spc.llm.requests.post", return_value=show) as post:
            self.assertEqual(llm.capacity("ollama-qwen3-8b"), (40960, "model"))
            self.assertEqual(llm.capacity("ollama-qwen3-8b"), (40960, "model"))
        self.assertEqual(post.call_count, 1)                                                  # kept 10 minutes
        listed = MagicMock(json=MagicMock(return_value={"data": [{"id": "qwen38-27b-fp8", "max_model_len": 262144}]}))
        with override_settings(DGX_CHAT_MODEL="qwen38-27b-fp8"), patch("spc.llm.requests.get", return_value=listed):
            self.assertEqual(llm.capacity("dgx-qwen38-27b-fp8"), (262144, "model"))
        with patch("spc.llm.requests.post", side_effect=requests.ConnectionError("down")):
            llm._capacity_cache.clear()
            self.assertEqual(llm.capacity("ollama-qwen3-8b"), (40960, "setting"))
        self.assertEqual(llm.capacity("external-gpt"), (131072, "setting"))


class CheckTests(SimpleTestCase):
    def test_every_number_of_the_text_is_looked_up_at_the_precision_it_is_given(self):
        answer = ("UCL 100.03 and LCL 79.81 (79.806 rounded), Cpk 2.03, sigma shift 0.06 σ, mean about 90. "
                  "A mean of 91.7 and a Cpk of 1.12 are not in the data. On 2026-10-13 at 21:50 the value read 3 times. "
                  "Capable means 1.33 or more. 1,229 points.\n```json\n{\"in_control\": true, \"value\": 555.5}\n```")
        found = llm.check_numbers(answer, PACKET | {"chart": {"points": 1229}})
        self.assertEqual([m["value"] for m in found["not_found"]], ["91.7", "1.12"])
        self.assertEqual((found["checked"], found["found"]), (9, 7))       # 100.03 79.81 79.806 2.03 0.06 90 91.7 1.12 1,229
        self.assertIn("A mean of 91.7", found["not_found"][0]["context"])
        self.assertEqual(found["not_checked"], 2)                           # 3 and 1.33; the date and the time are left out

    def test_a_share_may_be_written_as_a_percentage(self):
        self.assertEqual(llm.check_numbers("13.7 % of the values", {"share": 0.137})["not_found"], [])

    def test_the_verdict_is_read_and_compared_with_the_packet(self):
        answer = ("Text.\n```json\n{\"in_control\": true, \"capable\": true, \"events_seen\": 3, "
                  "\"patterns\": [{\"type\": \"stuck\", \"start\": \"2026-10-13T21:50:00Z\", \"end\": \"22:00\"}, "
                  "{\"type\": \"spike\", \"start\": \"08:15\", \"end\": null}]}\n```")
        result = llm.check_verdict(answer, PACKET)
        items = {i["item"]: (i["model"], i["data"], i["agree"]) for i in result["items"]}
        self.assertEqual(items, {"In control": (True, False, False),          # 2 points beyond the control limits
                                 "Capable (Cpk ≥ 1.33)": (True, True, True), "SPC events": (3, 3, True)})
        self.assertEqual([p["type"] for p in result["patterns"]], ["stuck", "spike"])
        self.assertEqual(result["times_outside_window"], ["08:15"])           # 21:50 and 22:00 are in the window

    def test_no_verdict_and_no_capability(self):
        self.assertEqual(llm.check_verdict("No block here.", PACKET)["found"], False)
        no_cpk = llm.check_verdict("```json\n{\"in_control\": false, \"capable\": null, \"events_seen\": 2}\n```",
                                   PACKET | {"capability": {"cpk": None}})
        self.assertEqual([(i["data"], i["agree"]) for i in no_cpk["items"]], [(False, True), (None, None), (3, False)])

    def test_the_packet_never_holds_the_injected_anomaly_record(self):
        text = (Path(llm.__file__)).read_text()
        self.assertNotIn("LiveSimEvent", text)
        self.assertNotIn("from historian.models", text)


@override_settings(SPC_LLM_NUM_CTX=16384, SPC_LLM_INPUT_LIMIT=16384, SPC_LLM_OUTPUT_LIMIT=4096, SPC_LLM_TEMPERATURE=0.2,
                   OLLAMA_BASE_URL="http://ollama:11434",
                   DGX_API_BASE="http://dgx:8004", DGX_CHAT_MODEL="qwen38", GROK_API_KEY="")
class LlmApiTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        caps = {"ollama-qwen3-8b": (40960, "model"), "dgx-qwen38-27b-fp8": (262144, "model"), "external-gpt": (131072, "setting")}
        patch("spc.llm.capacity", side_effect=lambda model_id: caps[model_id]).start()     # the models are not asked here
        self.tool_ = self.tool(usl=11.0, rules=["beyond_limits", "spec", "no_variation"])
        self.special.update({at(5): 10.5, at(5, 20): 10.6, at(8): 11.5})
        self.run_pass()                                     # the evaluator wrote two events

    def get(self):
        return self.client.get(f"/api/spc/tools/{self.tool_.id}/llm/").json()

    def post(self, **body):
        return self.client.post(f"/api/spc/tools/{self.tool_.id}/llm/analysis/", body, format="json")

    def test_the_packet_has_the_tool_its_limits_the_chart_the_distribution_capability_and_events(self):
        body = self.get()
        p = body["packet"]
        self.assertEqual(body["prompt"], llm.DEFAULT_PROMPT)
        self.assertEqual(body["default_model"], "ollama-qwen3-8b")
        self.assertEqual([(m["id"], m["capacity"], m["capacity_from"]) for m in body["models"]],
                         [("ollama-qwen3-8b", 40960, "model"), ("dgx-qwen38-27b-fp8", 262144, "model"), ("external-gpt", 131072, "setting")])
        self.assertEqual(body["limits"], {"input": 16384, "output": 4096})
        self.assertGreater(body["approx_tokens"], body["characters"] // 2)              # the prompt is counted too
        self.assertEqual((p["tool"]["name"], p["tool"]["chart"], p["tool"]["specification"]["usl"]), ("Product moisture", "I-MR", 11.0))
        self.assertEqual(p["tool"]["rules_on"], ["beyond_limits", "spec", "no_variation"])
        self.assertEqual((p["window"]["end"], p["window"]["hours"]), ("2026-10-12T12:10:00Z", 8))
        self.assertEqual([(c["cl"], c["lcl"], c["ucl"], c["source"]) for c in p["control_limits"]], [(9.5, 9.0, 10.0, "fixed")])
        self.assertEqual(p["chart"]["fired_by_rule"], {"beyond_limits": 3, "spec": 1})
        self.assertEqual([f["ts"] for f in p["chart"]["fired_points"]], ["2026-10-12T12:05:00Z", "2026-10-12T12:05:20Z", "2026-10-12T12:08:00Z"])
        self.assertEqual(p["events"]["in_window"], SpcEvent.objects.count())
        self.assertEqual(sum(t["n"] for t in p["trend"]), p["chart"]["points"])
        self.assertEqual(p["distribution"]["max"], 11.5)
        self.assertIn("not_available_because", p["capability"])

    def test_one_call_to_ollama_with_the_context_sent_and_thinking_off_and_the_checks(self):
        p = self.get()["packet"]
        answer = "Points beyond the UCL 10.0: 3. A mean of 12.34.\n```json\n{\"in_control\": false, \"capable\": null, \"events_seen\": 2}\n```"
        with patch("spc.llm.requests.post", return_value=ollama_reply(answer)) as post:
            r = self.post(model="ollama-qwen3-8b", prompt="My own prompt", packet=p)
        self.assertEqual(r.status_code, 200)
        url, sent = post.call_args.args[0], post.call_args.kwargs["json"]
        self.assertEqual(url, "http://ollama:11434/api/chat")
        self.assertEqual((sent["model"], sent["think"], sent["stream"]), ("qwen3:8b", False, False))
        self.assertEqual(sent["options"], {"num_ctx": 20480, "temperature": 0.2, "num_predict": 4096})   # room for both limits
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])          # no history: one call
        self.assertEqual(sent["messages"][0]["content"], "My own prompt")
        self.assertIn('"name":"Product moisture"', sent["messages"][1]["content"])
        body = r.json()
        self.assertEqual((body["input_tokens"], body["output_tokens"], body["tokens_estimated"], body["prompt_changed"]), (5098, 412, False, True))
        self.assertEqual((body["finished"], body["cut"]), ("stop", False))
        self.assertEqual({k: body["settings"][k] for k in ("input_limit", "output_limit", "capacity", "context_tokens")},
                         {"input_limit": 16384, "output_limit": 4096, "capacity": 40960, "context_tokens": 20480})
        self.assertEqual(body["timing"], {"load_seconds": 5.54, "input_seconds": 1.24, "output_seconds": 4.62,
                                          "input_tokens_per_second": 4111.3, "output_tokens_per_second": 89.2})
        self.assertEqual([m["value"] for m in body["checks"]["numbers"]["not_found"]], ["12.34"])
        # "capable": null while the data has a Cpk (one-sided, USL only) is a disagreement
        self.assertEqual([i["agree"] for i in body["checks"]["verdict"]["items"]], [True, False, True])

        from usage.models import ModelCall
        call = ModelCall.objects.get()
        self.assertEqual((call.purpose, call.model_id, call.prompt_tokens, call.completion_tokens), ("spc", "ollama-qwen3-8b", 5098, 412))

    def test_the_dgx_is_called_with_thinking_off_and_its_thinking_is_never_shown(self):
        reply = MagicMock()
        reply.json.return_value = {"choices": [{"message": {"content": "<think>hmm</think>The answer."}}],
                                   "usage": {"prompt_tokens": 4800, "completion_tokens": 300}}
        with patch("spc.llm.requests.post", return_value=reply) as post:
            body = self.post(model="dgx-qwen38-27b-fp8", prompt=llm.DEFAULT_PROMPT, packet=self.get()["packet"]).json()
        sent = post.call_args.kwargs["json"]
        self.assertEqual((sent["model"], sent["chat_template_kwargs"], sent["max_tokens"]), ("qwen38", {"enable_thinking": False}, 4096))
        self.assertEqual((body["answer"], body["prompt_changed"], body["input_tokens"]), ("The answer.", False, 4800))
        self.assertEqual(body["checks"]["verdict"]["found"], False)

    def test_the_run_has_an_input_and_an_output_limit_up_to_the_models_capacity(self):
        p = self.get()["packet"]
        with patch("spc.llm.requests.post", return_value=ollama_reply(done_reason="length")) as post:
            body = self.post(model="ollama-qwen3-8b", prompt="x", packet=p, input_limit=8192, output_limit=512).json()
        options = post.call_args.kwargs["json"]["options"]
        self.assertEqual((options["num_ctx"], options["num_predict"]), (16384, 512))         # the app's 16K is enough: no reload
        self.assertEqual((body["settings"]["input_limit"], body["settings"]["output_limit"], body["cut"]), (8192, 512, True))
        with patch("spc.llm.requests.post", return_value=ollama_reply()) as post:
            self.post(model="ollama-qwen3-8b", prompt="x", packet=p, input_limit=30000, output_limit=10960)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], 40960)          # both limits fit: the whole capacity

        refused = {
            "over the capacity": dict(input_limit=30000, output_limit=12000),
            "input over its limit": dict(input_limit=1000, output_limit=512),
            "not a number": dict(input_limit="many", output_limit=512),
            "too small": dict(input_limit=8192, output_limit=10),
        }
        with patch("spc.llm.requests.post") as post:
            answers = {name: self.post(model="ollama-qwen3-8b", prompt="x", packet=p, **limits) for name, limits in refused.items()}
            post.assert_not_called()                                                          # refused before any call
        self.assertEqual({name: r.status_code for name, r in answers.items()}, dict.fromkeys(refused, 400))
        self.assertIn("pass the model's capacity of 40,960 tokens", answers["over the capacity"].json()["error"])
        self.assertIn("above the input limit of 1,000", answers["input over its limit"].json()["error"])
        with patch("spc.llm.requests.post", return_value=MagicMock(json=MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]}))) as post:
            self.post(model="dgx-qwen38-27b-fp8", prompt="x", packet=p, input_limit=100000, output_limit=20000)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 20000)                  # the DGX holds 262,144

    def test_a_model_that_does_not_answer_or_is_not_offered_is_said(self):
        p = self.get()["packet"]
        with patch("spc.llm.requests.post", side_effect=requests.Timeout("slow")):
            r = self.post(model="ollama-qwen3-8b", prompt="x", packet=p)
        self.assertEqual((r.status_code, r.json()["error"]), (504, "ollama-qwen3-8b did not answer within 240 s"))
        with patch("spc.llm.requests.post", side_effect=requests.ConnectionError("refused")):
            self.assertEqual(self.post(model="ollama-qwen3-8b", prompt="x", packet=p).status_code, 502)
        self.assertEqual(self.post(model="external-gpt", prompt="x", packet=p).status_code, 502)       # no Grok key
        self.assertEqual(self.post(model="local-small", prompt="x", packet=p).status_code, 400)
        self.assertEqual(self.post(model="ollama-qwen3-8b", prompt="  ", packet=p).status_code, 400)
        self.assertEqual(self.post(model="ollama-qwen3-8b", prompt="x", packet={"a": 1}).status_code, 400)
        self.assertEqual(self.client.get("/api/spc/tools/00000000-0000-0000-0000-000000000000/llm/").status_code, 404)

    def test_nothing_is_stored_but_the_usage_record(self):
        before = (SpcEvent.objects.count(), self.tool_.version)
        with patch("spc.llm.requests.post", return_value=ollama_reply()):
            self.post(model="ollama-qwen3-8b", prompt="x", packet=self.get()["packet"])
            self.post(model="ollama-qwen3-8b", prompt="x", packet=self.get()["packet"])
        self.tool_.refresh_from_db()
        self.assertEqual((SpcEvent.objects.count(), self.tool_.version), before)
