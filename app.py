"""Assistant Randoneo — l'application terminée du chapitre précédent.

Interface Streamlit sur un agent outillé par son serveur MCP : historique en
session, agent et boucle d'événements en cache, étapes et tokens diffusés,
validation humaine sur l'ouverture de ticket, accès fermé par mot de passe.
Ce fichier est fourni tel quel : ce lab ne change pas une ligne de l'application.

Hors conteneur, depuis CE dossier :  streamlit run app.py
"""
import asyncio
import hmac
import queue
import threading
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import build_agent

st.set_page_config(page_title="Assistant Randoneo", layout="centered")


def check_access() -> bool:
    """Rend True si la session est déjà validée ; sinon affiche la porte."""
    if st.session_state.get("access_granted"):
        return True
    st.title("Assistant Randoneo")
    typed = st.text_input("Mot de passe d'accès", type="password")
    # compare_digest refuse deux `str` non-ASCII : on compare des octets, sinon
    # un mot de passe accentué lève un TypeError en pleine page.
    if typed and hmac.compare_digest(typed.encode(), st.secrets["app_password"].encode()):
        st.session_state["access_granted"] = True
        st.rerun()  # sans cette relance, la porte reste affichée au-dessus de l'app
    if typed:
        st.error("Mot de passe incorrect.")
    return False


if not check_access():
    st.stop()

DONE = object()  # sentinelle de fin de flux

st.title("Assistant Randoneo")
st.caption("Support client — agent outillé")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Compté en tête : une exécution qui se termine par st.rerun() n'atteint
# jamais la fin du fichier.
st.session_state.reruns = st.session_state.get("reruns", 0) + 1

if "pending" not in st.session_state:
    st.session_state.pending = None

if "thread_id" not in st.session_state:
    # Un fil par session : deux onglets ne partagent pas la même conversation.
    st.session_state.thread_id = uuid.uuid4().hex[:8]

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}


@st.cache_resource
def get_runtime():
    """La boucle d'événements de l'application, et l'agent qui vit dessus."""
    st.session_state.agent_builds = st.session_state.get("agent_builds", 0) + 1
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    agent = asyncio.run_coroutine_threadsafe(build_agent(), loop).result()
    return loop, agent


loop, agent = get_runtime()


def format_step(call) -> str:
    """Une étape lisible plutôt qu'une trace brute."""
    args = " · ".join(f"{name}={value}" for name, value in call["args"].items()
                      if value is not None)
    return f"Outil {call['name']} — {args}" if args else f"Outil {call['name']}"


def render(message):
    """Rejoue un message de l'historique, ses étapes comprises."""
    with st.chat_message(message["role"]):
        for step in message.get("steps", []):
            st.caption(step)
        if message["content"]:
            st.markdown(message["content"])


def run_turn(payload):
    """Joue un tour d'agent et l'affiche au fil de l'eau."""
    events: queue.Queue = queue.Queue()

    async def pump():
        try:
            async for event in agent.astream(payload, CONFIG,
                                             stream_mode=["updates", "messages"]):
                events.put(event)
        except Exception as error:
            events.put(error)
        finally:
            events.put(DONE)

    asyncio.run_coroutine_threadsafe(pump(), loop)

    answer, steps, pending, box, shown = "", [], None, None, ""
    while True:
        event = events.get()
        if event is DONE:
            break
        if isinstance(event, Exception):
            st.error(f"L'agent a échoué : {event}")
            return answer, steps, None
        mode, data = event
        if mode == "updates":
            for key, update in data.items():
                if key == "__interrupt__":
                    pending = update[0].value
                    continue
                if not isinstance(update, dict):
                    continue
                message = (update.get("messages") or [None])[-1]
                for call in getattr(message, "tool_calls", None) or []:
                    step = format_step(call)
                    steps.append(step)
                    st.caption(step)
                    box = None  # le texte qui suit ira sous l'étape
        else:
            chunk, _ = data
            if chunk.type == "AIMessageChunk" and chunk.text:
                if box is None:
                    box, shown = st.empty(), ""
                answer += chunk.text
                shown += chunk.text
                box.markdown(shown)

    # Filet : rien ne garantit qu'un graphe repris rediffuse des tokens.
    if not answer and pending is None:
        state = asyncio.run_coroutine_threadsafe(agent.aget_state(CONFIG), loop).result()
        answer = next((m.text for m in reversed(state.values["messages"])
                        if m.type in ("ai", "tool") and m.text), "")
        if answer:
            st.markdown(answer)

    return answer, steps, pending


# 1. Rejouer l'historique, puisque le rerun a effacé la page.
for message in st.session_state.messages:
    render(message)

# 2. Traiter le nouveau message, ou la décision prise au rerun précédent.
question = st.chat_input("Posez votre question",
                         disabled=bool(st.session_state.pending))

payload = None
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render(st.session_state.messages[-1])
    payload = {"messages": [HumanMessage(question)]}
elif "decision" in st.session_state:
    payload = Command(resume=st.session_state.pop("decision"))

if payload is not None:
    with st.chat_message("assistant"):
        answer, steps, pending = run_turn(payload)
    if answer or steps:
        st.session_state.messages.append({"role": "assistant", "content": answer,
                                          "steps": steps})
    if pending:
        st.session_state.pending = pending
        st.rerun()

# 3. La carte de validation, quand une action attend une décision.
if st.session_state.pending:
    action = st.session_state.pending
    with st.chat_message("assistant"):
        st.warning("L'agent demande l'autorisation d'ouvrir un ticket au support.")
        st.markdown(
            f"- Motif : `{action['intent']}`\n"
            f"- Résumé : {action['summary']}\n"
            f"- Commande : {action['order_id'] or 'aucune'}"
        )
        approve, reject = st.columns(2)
        if approve.button("Approuver", type="primary"):
            st.session_state.decision = "approve"
            st.session_state.pending = None
            st.rerun()
        if reject.button("Refuser"):
            st.session_state.decision = "reject"
            st.session_state.pending = None
            st.rerun()

# Le bandeau de mesure est en fin de fichier : le bouton y est donc traité après
# la boucle d'affichage, ce qui rend `st.rerun()` réellement nécessaire.
with st.sidebar:
    st.subheader("Mesures")
    st.caption(f"Réexécutions du script : {st.session_state.reruns}")
    st.caption(f"Constructions de l'agent : {st.session_state.get('agent_builds', 0)}")
    st.caption(f"Fil de conversation : {st.session_state.thread_id}")
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.rerun()
