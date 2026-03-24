from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, GraphOutput, Interrupt, interrupt

HIGH_RISK_LEVELS = {"high", "refund", "escalation"}
DEFAULT_DB_DIR = Path.cwd() / ".langgraph-demo"
DEFAULT_CHECKPOINT_PATH = DEFAULT_DB_DIR / "customer_support.sqlite"
DEFAULT_MEMORY_PATH = DEFAULT_DB_DIR / "customer_support_memories.json"


class DraftAction(TypedDict):
    type: str
    summary: str
    customer_reply: str
    requires_human: bool
    reason: str


class ClassificationResult(TypedDict):
    intent: str
    risk_level: str
    reason: str


class FinalReplyResult(TypedDict):
    final_response: str
    required_documents: list[str]


class PersistedMemory(TypedDict):
    namespace: list[str]
    key: str
    value: dict[str, Any]


class HumanDecision(TypedDict):
    type: str
    text: NotRequired[str]


class CustomerSupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    ticket_id: str
    intent: str
    risk_level: str
    classification_reason: str
    draft_action: DraftAction
    approval_notes: str
    human_instructions: str
    review_decision: str
    final_response: str
    action_result: str
    required_documents: list[str]
    memory_hits: list[str]
    loaded_memory_keys: list[str]


class InteractiveHumanDecision(TypedDict):
    type: str
    text: str


class SupportDemoModel(Protocol):
    def classify_ticket(
        self, *, message: str, memory_hits: Sequence[str]
    ) -> ClassificationResult: ...

    def draft_resolution(
        self,
        *,
        message: str,
        intent: str,
        risk_level: str,
        memory_hits: Sequence[str],
        approval_notes: str | None,
    ) -> DraftAction: ...

    def finalize_reply(
        self,
        *,
        message: str,
        action: DraftAction,
        action_result: str,
        memory_hits: Sequence[str],
        human_instructions: str | None,
    ) -> FinalReplyResult: ...


class HeuristicSupportDemoModel:
    # 仅用于测试或显式调试的兜底模型。
    # 它和真实 LLM 路径保持同样的两个决策点：
    # 1. 分类工单
    # 2. 起草处理动作
    def classify_ticket(
        self, *, message: str, memory_hits: Sequence[str]
    ) -> ClassificationResult:
        text = message.lower()
        profile_text = " ".join(memory_hits).lower()

        if "退款" in message or "refund" in text:
            return {
                "intent": "refund_request",
                "risk_level": "refund",
                "reason": "Refunds move money and require manual approval in this demo.",
            }
        if (
            "投诉" in message
            or "经理" in message
            or "escalat" in text
            or "supervisor" in text
        ):
            return {
                "intent": "escalation_request",
                "risk_level": "escalation",
                "reason": "Escalations need a person to review tone and remedy.",
            }
        if "vip" in text or "高价值" in profile_text:
            return {
                "intent": "priority_shipping",
                "risk_level": "high",
                "reason": "High-value users trigger a manual quality check.",
            }
        if "优惠券" in message or "coupon" in text:
            return {
                "intent": "coupon_request",
                "risk_level": "low",
                "reason": "Coupon requests can be handled automatically.",
            }
        return {
            "intent": "shipping_follow_up",
            "risk_level": "low",
            "reason": "General support request with low operational risk.",
        }

    def draft_resolution(
        self,
        *,
        message: str,
        intent: str,
        risk_level: str,
        memory_hits: Sequence[str],
        approval_notes: str | None,
    ) -> DraftAction:
        language_hint = (
            "中文" if any("中文" in item for item in memory_hits) else "English"
        )
        note_suffix = f" Human note: {approval_notes}" if approval_notes else ""
        lowered_notes = approval_notes.lower() if approval_notes else ""

        if intent == "refund_request":
            if approval_notes and (
                "拒绝" in approval_notes or "不退款" in approval_notes
            ):
                return {
                    "type": "reply_only",
                    "summary": "Do not grant the refund; explain the next review steps."
                    + note_suffix,
                    "customer_reply": (
                        "这次暂时不能直接退款，但我会继续协助你完成后续核验流程。"
                        if language_hint == "中文"
                        else "We cannot issue an immediate refund right now, but I will help with the next review steps."
                    ),
                    "requires_human": False,
                    "reason": "Human policy asked to decline the refund.",
                }
            if approval_notes and (
                "优惠券" in approval_notes or "coupon" in lowered_notes
            ):
                return {
                    "type": "coupon",
                    "summary": "Offer a coupon instead of a direct refund."
                    + note_suffix,
                    "customer_reply": (
                        "这次先为你提供补偿优惠券，我也会继续跟进后续处理。"
                        if language_hint == "中文"
                        else "I can offer a courtesy coupon first and continue following up on the case."
                    ),
                    "requires_human": False,
                    "reason": "Human policy replaced refund with coupon.",
                }
            return {
                "type": "refund",
                "summary": "Offer a full refund after human approval." + note_suffix,
                "customer_reply": (
                    "可以为你处理退款，我先提交给人工主管做最终确认。"
                    if language_hint == "中文"
                    else "I can process a refund, but it needs a final human approval first."
                ),
                "requires_human": True,
                "reason": "Monetary action.",
            }
        if intent == "escalation_request":
            return {
                "type": "escalate",
                "summary": "Escalate the ticket to a human support lead." + note_suffix,
                "customer_reply": (
                    "我会升级给人工主管继续跟进。"
                    if language_hint == "中文"
                    else "I will escalate this to a human support lead."
                ),
                "requires_human": True,
                "reason": "Escalation requested.",
            }
        if intent == "priority_shipping":
            return {
                "type": "priority_shipping",
                "summary": "Upgrade the shipping priority and apologize." + note_suffix,
                "customer_reply": (
                    "我可以帮你提升处理优先级，并同步物流进展。"
                    if language_hint == "中文"
                    else "I can raise the shipping priority and follow up on delivery."
                ),
                "requires_human": risk_level in HIGH_RISK_LEVELS,
                "reason": "High-value user experience.",
            }
        if intent == "coupon_request":
            return {
                "type": "coupon",
                "summary": "Issue a service recovery coupon." + note_suffix,
                "customer_reply": (
                    "我会给你补发一张优惠券，并继续跟进后续体验。"
                    if language_hint == "中文"
                    else "I will issue a courtesy coupon and continue to monitor the case."
                ),
                "requires_human": False,
                "reason": "Low-risk service recovery.",
            }
        return {
            "type": "reply_only",
            "summary": "Send a status update with empathy and next steps."
            + note_suffix,
            "customer_reply": (
                "我已经记录你的问题，会继续帮你跟进配送进展。"
                if language_hint == "中文"
                else "I have logged the issue and will keep tracking the shipment for you."
            ),
            "requires_human": False,
            "reason": "Informational response only.",
        }

    def finalize_reply(
        self,
        *,
        message: str,
        action: DraftAction,
        action_result: str,
        memory_hits: Sequence[str],
        human_instructions: str | None,
    ) -> FinalReplyResult:
        required_documents: list[str] = []
        if human_instructions and (
            "证件" in human_instructions
            or "材料" in human_instructions
            or "证明" in human_instructions
            or "document" in human_instructions.lower()
        ):
            required_documents = ["身份证明", "订单截图"]
        suffix = ""
        if required_documents:
            suffix = f" 请准备以下材料：{'、'.join(required_documents)}。"
        if human_instructions and (
            "拒绝" in human_instructions or "不退款" in human_instructions
        ):
            final_response = (
                f"这次暂时无法直接退款，但我们会继续协助处理。 {action_result}{suffix}"
            )
        else:
            final_response = (
                f"{action['customer_reply']} 操作结果：{action_result}{suffix}"
            )
        if not any("中文" in item for item in memory_hits):
            docs_text = (
                f" Please prepare: {', '.join(required_documents)}."
                if required_documents
                else ""
            )
            if human_instructions and (
                "拒绝" in human_instructions
                or "not refund" in human_instructions.lower()
            ):
                final_response = (
                    "We cannot issue an immediate refund for now, but we will keep helping."
                    f" Result: {action_result}{docs_text}"
                )
            else:
                final_response = (
                    f"{action['customer_reply']} Result: {action_result}{docs_text}"
                )
        return {
            "final_response": final_response,
            "required_documents": required_documents,
        }


class LangChainStructuredSupportModel:
    def __init__(self, model: Any):
        self._model = model

    def classify_ticket(
        self, *, message: str, memory_hits: Sequence[str]
    ) -> ClassificationResult:
        # 第一步 LLM 决策：判断用户意图和风险等级。
        structured = self._model.with_structured_output(ClassificationResult)
        prompt = (
            "Classify this customer-support ticket for a LangGraph demo. "
            "Return JSON with intent, risk_level, and reason.\n"
            f"Memory hits: {list(memory_hits)}\n"
            f"Message: {message}"
        )
        return structured.invoke(prompt)

    def finalize_reply(
        self,
        *,
        message: str,
        action: DraftAction,
        action_result: str,
        memory_hits: Sequence[str],
        human_instructions: str | None,
    ) -> FinalReplyResult:
        structured = self._model.with_structured_output(FinalReplyResult)
        prompt = (
            "You are finalizing a customer-support response in a LangGraph demo. "
            "Return JSON with final_response and required_documents.\n"
            f"Original user message: {message}\n"
            f"Chosen action: {action}\n"
            f"Execution result: {action_result}\n"
            f"Memory hits: {list(memory_hits)}\n"
            f"Human instructions: {human_instructions}\n"
            "If the human asks to change refusal style, follow that instruction. "
            "If the human asks to request documents on success, include them in "
            "`required_documents` and mention them in `final_response`."
        )
        return structured.invoke(prompt)

    def draft_resolution(
        self,
        *,
        message: str,
        intent: str,
        risk_level: str,
        memory_hits: Sequence[str],
        approval_notes: str | None,
    ) -> DraftAction:
        # 第二步 LLM 决策：给出 agent 想执行的具体动作。
        structured = self._model.with_structured_output(DraftAction)
        prompt = (
            "Draft a safe customer-support action for a LangGraph demo. "
            "Return JSON with type, summary, customer_reply, requires_human, and reason.\n"
            f"Intent: {intent}\n"
            f"Risk: {risk_level}\n"
            f"Approval notes: {approval_notes}\n"
            f"Memory hits: {list(memory_hits)}\n"
            f"Message: {message}"
        )
        return structured.invoke(prompt)


def build_default_model() -> SupportDemoModel:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    allow_heuristic = os.getenv("LANGGRAPH_DEMO_ALLOW_HEURISTIC", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if api_key:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": api_key,
                "temperature": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            return LangChainStructuredSupportModel(ChatOpenAI(**kwargs))
        except ImportError as exc:
            if not allow_heuristic:
                raise RuntimeError(
                    "The customer-support demo now requires a real LLM by default. "
                    "Install `langchain-openai` and configure `OPENAI_API_KEY`, or set "
                    "`LANGGRAPH_DEMO_ALLOW_HEURISTIC=true` to use the local fallback model."
                ) from exc
    if allow_heuristic:
        return HeuristicSupportDemoModel()
    raise RuntimeError(
        "The customer-support demo requires a real LLM by default. "
        "Set `OPENAI_API_KEY` and install `langchain-openai`, or explicitly opt in to "
        "the local fallback with `LANGGRAPH_DEMO_ALLOW_HEURISTIC=true`."
    )


def build_graph(
    checkpointer: BaseCheckpointSaver, store: BaseStore, model: SupportDemoModel | Any
) -> Any:
    support_model = _coerce_model(model)

    def load_profile_memory(
        state: CustomerSupportState, config: RunnableConfig, store: BaseStore
    ) -> dict[str, Any]:
        # 长期记忆不放在 thread state 里，而是放在 store 中。
        # 每次运行开始时，先把用户画像/历史加载到当前工作状态，
        # 这样后续节点里的 LLM 才能利用这些信息做判断。
        user_id = state["user_id"]
        _ensure_seed_memories(store, user_id)
        items = store.search(_memory_namespace(user_id), limit=20)
        memory_hits = [str(item.value["text"]) for item in items]
        loaded_memory_keys = [item.key for item in items]
        return {
            "memory_hits": memory_hits,
            "loaded_memory_keys": loaded_memory_keys,
            "ticket_id": state.get("ticket_id") or _ticket_id(config),
        }

    def classify_ticket(state: CustomerSupportState) -> dict[str, Any]:
        # LangGraph 节点 1：调用 LLM 对当前工单做分类。
        message = _last_human_message(state["messages"])
        result = support_model.classify_ticket(
            message=message,
            memory_hits=state.get("memory_hits", []),
        )
        return {
            "intent": result["intent"],
            "risk_level": result["risk_level"],
            "classification_reason": result["reason"],
        }

    def draft_resolution(state: CustomerSupportState) -> dict[str, Any]:
        # LangGraph 节点 2：调用 LLM 起草处理方案。
        message = _last_human_message(state["messages"])
        draft = support_model.draft_resolution(
            message=message,
            intent=state["intent"],
            risk_level=state["risk_level"],
            memory_hits=state.get("memory_hits", []),
            approval_notes=state.get("approval_notes"),
        )
        return {"draft_action": draft}

    def human_review(state: CustomerSupportState) -> dict[str, Any]:
        # 人工介入发生在这里。
        #
        # `interrupt(...)` 的含义不是“在这个函数里等待人工输入”，
        # 而是“立即暂停整条 graph，把控制权还给外部调用方”。
        #
        # 也就是说这条链路会分成两段：
        # 1. `start` 运行到这里，返回 `status="interrupted"`
        # 2. 人工在 graph 外查看 payload 并做决定
        # 3. 再由外部调用 `Command(resume=...)` 继续同一个 thread
        payload = {
            "ticket_id": state["ticket_id"],
            "user_id": state["user_id"],
            "ticket_summary": _last_human_message(state["messages"]),
            "risk_level": state["risk_level"],
            "risk_reason": state["classification_reason"],
            "suggested_action": state["draft_action"],
            "memory_hits": state.get("memory_hits", []),
            "editable_fields": ["draft_action.summary", "draft_action.customer_reply"],
        }
        decision = interrupt(payload)
        return _apply_human_decision(state, decision)

    def apply_human_policy(state: CustomerSupportState) -> dict[str, Any]:
        # 人工可以通过全局指令影响后续 agent 决策，而不只是修改一句回复。
        # 当 review_decision=response 时，我们把人工指令重新喂给 LLM，
        # 让它基于人工政策再起草一次动作。
        if state.get("review_decision") != "response":
            return {}
        message = _last_human_message(state["messages"])
        draft = support_model.draft_resolution(
            message=message,
            intent=state["intent"],
            risk_level=state["risk_level"],
            memory_hits=state.get("memory_hits", []),
            approval_notes=state.get("human_instructions"),
        )
        return {"draft_action": draft}

    def execute_action(state: CustomerSupportState) -> dict[str, Any]:
        # 这里模拟真正的业务副作用。
        # 真正执行前，策略已经由 LLM 决定；高风险动作还会先经过人工审批。
        action = state["draft_action"]
        action_type = action["type"]
        if action_type == "refund":
            action_result = "Refund approved and queued for finance."
        elif action_type == "coupon":
            action_result = "Coupon issued to the customer."
        elif action_type == "priority_shipping":
            action_result = "Priority shipping escalation sent to operations."
        elif action_type == "escalate":
            action_result = "Ticket escalated to a human support lead."
        else:
            action_result = "Response sent without an operational side effect."
        return {"action_result": action_result}

    def persist_new_memory(
        state: CustomerSupportState, store: BaseStore
    ) -> dict[str, Any]:
        # 把这次工单结果写回长期记忆。
        # 这样同一个用户下一次开启新 thread 时，也能回忆起这次历史。
        memory_text = (
            f"Ticket {state['ticket_id']} ended with action '{state['draft_action']['type']}'. "
            f"Customer asked: {_last_human_message(state['messages'])}"
        )
        store.put(
            _memory_namespace(state["user_id"]),
            f"ticket-{state['ticket_id']}",
            {
                "text": memory_text,
                "kind": "ticket_outcome",
                "importance": 0.8 if state["risk_level"] in HIGH_RISK_LEVELS else 0.4,
            },
        )
        updated_hits = state.get("memory_hits", []) + [memory_text]
        return {"memory_hits": updated_hits}

    def finalize_reply(state: CustomerSupportState) -> dict[str, Any]:
        # 最终回复也交给模型/策略层，确保人工全局指令能影响拒绝话术、
        # 成功后的材料要求等。
        result = support_model.finalize_reply(
            message=_last_human_message(state["messages"]),
            action=state["draft_action"],
            action_result=state["action_result"],
            memory_hits=state.get("memory_hits", []),
            human_instructions=state.get("human_instructions"),
        )
        return {
            "final_response": result["final_response"],
            "required_documents": result["required_documents"],
            "messages": [AIMessage(content=result["final_response"])],
        }

    # LangGraph 拓扑：
    # 读长期记忆 -> LLM 分类 -> LLM 起草方案 -> （必要时）人工审批 ->
    # 根据人工策略重新规划 -> 执行动作 -> 写回长期记忆 -> 最终回复
    graph = StateGraph(CustomerSupportState)
    graph.add_node("load_profile_memory", load_profile_memory)
    graph.add_node("classify_ticket", classify_ticket)
    graph.add_node("draft_resolution", draft_resolution)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_human_policy", apply_human_policy)
    graph.add_node("execute_action", execute_action)
    graph.add_node("persist_new_memory", persist_new_memory)
    graph.add_node("finalize_reply", finalize_reply)
    graph.add_edge(START, "load_profile_memory")
    graph.add_edge("load_profile_memory", "classify_ticket")
    graph.add_edge("classify_ticket", "draft_resolution")
    graph.add_conditional_edges(
        "draft_resolution",
        _route_after_draft,
        {"human_review": "human_review", "execute_action": "execute_action"},
    )
    graph.add_edge("human_review", "apply_human_policy")
    graph.add_edge("apply_human_policy", "execute_action")
    graph.add_edge("execute_action", "persist_new_memory")
    graph.add_edge("persist_new_memory", "finalize_reply")
    graph.add_edge("finalize_reply", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def run_start(
    thread_id: str,
    user_id: str,
    message: str,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    model: SupportDemoModel | Any | None = None,
) -> dict[str, Any]:
    # 启动一次新的运行，输入是一条新的用户消息。
    # 如果 graph 在中途命中 `interrupt(...)`，这里返回的不是最终答案，
    # 而是一个“已暂停、等待人工处理”的 payload。这是预期行为。
    checkpoint_path = Path(checkpoint_path)
    memory_path = Path(memory_path)
    store = load_memory_store(memory_path)
    _ensure_seed_memories(store, user_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(checkpointer, store, model or build_default_model())
        config = _run_config(thread_id=thread_id, user_id=user_id, entrypoint="start")
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "ticket_id": _ticket_id(config),
            },
            config,
            version="v2",
        )
        payload = _graph_output_payload(graph, result, config)
    save_memory_store(store, memory_path)
    return payload


def run_resume(
    thread_id: str,
    human_decision: HumanDecision,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    model: SupportDemoModel | Any | None = None,
) -> dict[str, Any]:
    # 从 SQLite checkpoint 中恢复一次之前被中断的运行。
    checkpoint_path = Path(checkpoint_path)
    memory_path = Path(memory_path)
    store = load_memory_store(memory_path)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(checkpointer, store, model or build_default_model())
        config = _run_config(thread_id=thread_id, entrypoint="resume")
        result = graph.invoke(Command(resume=human_decision), config, version="v2")
        payload = _graph_output_payload(graph, result, config)
    save_memory_store(store, memory_path)
    return payload


def load_memory_store(path: str | Path) -> InMemoryStore:
    store = InMemoryStore()
    path = Path(path)
    if not path.exists():
        return store
    data = json.loads(path.read_text())
    for item in data:
        namespace = tuple(item["namespace"])
        store.put(namespace, item["key"], item["value"])
    return store


def save_memory_store(store: InMemoryStore, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[PersistedMemory] = []
    for namespace in store.list_namespaces(limit=10_000):
        for item in store.search(namespace, limit=10_000):
            serialized.append(
                {
                    "namespace": list(namespace),
                    "key": item.key,
                    "value": item.value,
                }
            )
    path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2, sort_keys=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-file LangGraph customer-support demo."
    )
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--thread-id", required=True)
    start_parser.add_argument("--user-id", required=True)
    start_parser.add_argument("--message", required=True)
    start_parser.add_argument(
        "--no-interactive-review",
        action="store_true",
        help="命中人工审批时不在控制台继续交互，直接打印 interrupt payload。",
    )

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--thread-id", required=True)
    decision_group = resume_parser.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--approve", action="store_true")
    decision_group.add_argument("--edit")
    decision_group.add_argument("--response")

    args = parser.parse_args(argv)

    if args.command == "start":
        payload = run_start(
            thread_id=args.thread_id,
            user_id=args.user_id,
            message=args.message,
            checkpoint_path=args.checkpoint_path,
            memory_path=args.memory_path,
        )
        if (
            not args.no_interactive_review
            and payload["status"] == "interrupted"
            and sys.stdin.isatty()
        ):
            payload = _run_interactive_review_loop(
                thread_id=args.thread_id,
                checkpoint_path=args.checkpoint_path,
                memory_path=args.memory_path,
            )
    else:
        payload = run_resume(
            thread_id=args.thread_id,
            human_decision=_decision_from_args(args),
            checkpoint_path=args.checkpoint_path,
            memory_path=args.memory_path,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _graph_output_payload(
    graph: Any, result: GraphOutput, config: RunnableConfig
) -> dict[str, Any]:
    state_snapshot = graph.get_state(config)
    payload = {
        "status": "interrupted" if result.interrupts else "completed",
        "interrupts": [_serialize_interrupt(item) for item in result.interrupts],
        "state": _serialize_state(state_snapshot.values),
        "next": list(state_snapshot.next),
        "checkpoint_count": len(list(graph.get_state_history(config))),
    }
    return payload


def _run_interactive_review_loop(
    *, thread_id: str, checkpoint_path: str, memory_path: str
) -> dict[str, Any]:
    # 在 IDE/终端里把 human-in-the-loop 直接做成交互式体验：
    # start 命中 interrupt 后，不让用户手动再敲一条 resume 命令，
    # 而是在当前控制台里读取人工决定，然后自动恢复 graph。
    payload = run_resume(
        thread_id=thread_id,
        human_decision=_prompt_human_decision(),
        checkpoint_path=checkpoint_path,
        memory_path=memory_path,
    )
    while payload["status"] == "interrupted":
        payload = run_resume(
            thread_id=thread_id,
            human_decision=_prompt_human_decision(),
            checkpoint_path=checkpoint_path,
            memory_path=memory_path,
        )
    return payload


def _route_after_draft(state: CustomerSupportState) -> str:
    # 只有高风险，或者明确要求人工确认的动作，才进入人工审批节点。
    if (
        state["risk_level"] in HIGH_RISK_LEVELS
        or state["draft_action"]["requires_human"]
    ):
        return "human_review"
    return "execute_action"


def _apply_human_decision(
    state: CustomerSupportState, decision: HumanDecision | dict[str, Any] | None
) -> dict[str, Any]:
    # 支持的人工输入：
    # - accept：直接批准
    # - edit：直接改写动作
    # - response：给一段反馈，让后续动作按反馈改写
    if not decision:
        return {
            "review_decision": "accept",
            "approval_notes": "Approved without edits.",
            "human_instructions": "",
        }

    decision_type = str(decision.get("type", "accept"))
    decision_text = str(decision.get("text", "")).strip()

    if decision_type == "accept":
        return {
            "review_decision": "accept",
            "approval_notes": "Approved without edits.",
            "human_instructions": "",
        }
    if decision_type == "edit":
        edited_action = {
            "type": "coupon",
            "summary": decision_text
            or "Manual edit: issue a coupon instead of refund.",
            "customer_reply": (
                decision_text or "已改为补偿优惠券方案，我会继续帮你跟进。"
            ),
            "requires_human": False,
            "reason": "Human edited the action.",
        }
        return {
            "review_decision": "edit",
            "approval_notes": decision_text or "Human edited the action.",
            "human_instructions": decision_text,
            "draft_action": edited_action,
        }
    if decision_type == "response":
        return {
            "review_decision": "response",
            "approval_notes": decision_text,
            "human_instructions": decision_text,
        }
    raise ValueError(f"Unsupported human decision type: {decision_type}")


def _coerce_model(model: SupportDemoModel | Any) -> SupportDemoModel:
    if hasattr(model, "classify_ticket") and hasattr(model, "draft_resolution"):
        return model
    return LangChainStructuredSupportModel(model)


def _ensure_seed_memories(store: BaseStore, user_id: str) -> None:
    namespace = _memory_namespace(user_id)
    if store.search(namespace, limit=5):
        return
    store.put(
        namespace,
        "profile-language",
        {"text": "用户偏好中文回复", "kind": "preference", "importance": 0.9},
    )
    store.put(
        namespace,
        "profile-shipping",
        {
            "text": "近 30 天多次催单，需要更积极同步物流进展",
            "kind": "history",
            "importance": 0.7,
        },
    )


def _memory_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "memories")


def _last_human_message(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("No human message found in state.")


def _run_config(
    *, thread_id: str, entrypoint: str, user_id: str | None = None
) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if user_id is not None:
        configurable["user_id"] = user_id
    return {
        "configurable": configurable,
        "metadata": {
            "demo": "customer_support",
            "entrypoint": entrypoint,
            "thread_id": thread_id,
        },
    }


def _ticket_id(config: RunnableConfig) -> str:
    return f"ticket-{config['configurable']['thread_id']}"


def _serialize_interrupt(item: Interrupt) -> dict[str, Any]:
    return {"id": item.id, "value": item.value}


def _serialize_state(state: CustomerSupportState) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if key == "messages":
            messages = cast(list[AnyMessage], value)
            serialized[key] = [_serialize_message(message) for message in messages]
        else:
            serialized[key] = value
    return serialized


def _serialize_message(message: AnyMessage) -> dict[str, Any]:
    return {
        "type": message.type,
        "content": message.content,
    }


def _prefers_chinese(memory_hits: Sequence[str]) -> bool:
    return any("中文" in item for item in memory_hits)


def _decision_from_args(args: argparse.Namespace) -> HumanDecision:
    if args.approve:
        return {"type": "accept"}
    if args.edit:
        return {"type": "edit", "text": args.edit}
    return {"type": "response", "text": args.response}


def _prompt_human_decision() -> HumanDecision:
    print("\n=== 人工审批 ===")
    print("输入 `approve` 直接批准")
    print("输入 `edit:你的改写内容` 直接改写动作")
    print("输入 `response:你的意见` 或 `policy:你的全局策略` 影响后续 agent 决策")
    while True:
        raw = input("审批输入> ").strip()
        if raw == "approve":
            return {"type": "accept"}
        if raw.startswith("edit:"):
            text = raw[len("edit:") :].strip()
            if text:
                return {"type": "edit", "text": text}
        if raw.startswith("response:"):
            text = raw[len("response:") :].strip()
            if text:
                return {"type": "response", "text": text}
        if raw.startswith("policy:"):
            text = raw[len("policy:") :].strip()
            if text:
                return {"type": "response", "text": text}
        print("无效输入，请重新输入。")


if __name__ == "__main__":
    raise SystemExit(main())
