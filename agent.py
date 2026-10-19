"""L'agent de support Randoneo : aucun outil n'est défini ici.

Les capacités viennent du serveur MCP, découvertes à la connexion ; l'agent
n'apporte que le raisonnement, la mémoire et le garde-fou humain sur l'action
sensible. Ce fichier est fourni tel quel : vous n'avez pas à le modifier.
"""
import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

MODEL = "claude-haiku-4-5"
SERVER = "randoneo"
SENSITIVE = "create_ticket"  # le seul tool qui écrit : soumis à validation

SYSTEM_PROMPT = (
    "Tu es l'assistant du support client de Randoneo, un vendeur de matériel de randonnée. "
    "Tu t'appuies sur tes outils : le suivi des commandes pour les données client, "
    "l'ouverture de ticket pour escalader vers un conseiller. "
    "Pour toute réclamation, tout litige ou toute demande de SAV, ouvre un ticket. "
    "Réponds en français, de façon concise, et n'invente jamais une information : "
    "si un outil ne la donne pas, dis-le."
)

# En stdio, c'est l'hôte qui lance le serveur : c'est donc lui qui décide de
# l'environnement du sous-processus. Le SDK MCP ne transmet pas le vôtre, il en
# fabrique un depuis une liste blanche (PATH, HOME, USER…) d'où toute clé d'API
# est absente. Et dès qu'on renseigne « env », ce dictionnaire REMPLACE cette
# liste blanche : ce qui n'y figure pas n'existe pas pour le serveur.
MCP_SERVERS = {
    SERVER: {
        "command": sys.executable,
        "args": ["randoneo_support.py"],
        "transport": "stdio",
        "env": {"PATH": os.environ["PATH"]},
        # Un serveur qui appelle une API tierce ajouterait sa clé ici, par ex. :
        # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    }
}

mcp_client = MultiServerMCPClient(MCP_SERVERS)


def _under_validation(mcp_tool):
    """Enveloppe le tool sensible d'un point d'arrêt humain.

    `interrupt()` est une primitive du graphe : elle ne peut pas vivre dans le
    serveur, qui tourne dans un autre processus. On garde donc l'action côté
    serveur et la décision côté agent — l'enveloppe demande le feu vert, puis délègue.
    """

    @tool(SENSITIVE, description=mcp_tool.description)
    async def validated(intent: str, summary: str, order_id: str | None = None) -> str:
        decision = interrupt({"action": SENSITIVE, "intent": intent,
                              "summary": summary, "order_id": order_id})
        if decision != "approve":
            return "Ouverture du ticket refusée par un opérateur."
        return await mcp_tool.ainvoke(
            {"intent": intent, "summary": summary, "order_id": order_id}
        )

    return validated


async def _hold_session(ready):
    """Tient la session MCP ouverte aussi longtemps que l'application vit.

    Une session ouverte dans une coroutine qui se termine est refermée avec elle :
    ses flux sont fermés et le premier appel d'outil échoue sur `ClosedResourceError`.
    On la garde donc dans une tâche qui ne se termine jamais — c'est elle qui tient
    le sous-processus du serveur, et qui évite de le relancer à chaque appel d'outil.
    """
    try:
        async with mcp_client.session(SERVER) as session:
            mcp_tools = await load_mcp_tools(session)

            by_name = {mcp_tool.name: mcp_tool for mcp_tool in mcp_tools}
            tools = [mcp_tool for mcp_tool in mcp_tools if mcp_tool.name != SENSITIVE]
            tools.append(_under_validation(by_name[SENSITIVE]))

            model = init_chat_model(MODEL, model_provider="anthropic",
                                    temperature=0, max_retries=8)
            ready.set_result(create_agent(model, tools, system_prompt=SYSTEM_PROMPT,
                                          checkpointer=InMemorySaver()))
            await asyncio.Event().wait()
    except Exception as error:  # le serveur n'a pas démarré : ne pas attendre indéfiniment
        if not ready.done():
            ready.set_exception(error)
        else:
            # La session est tombée après coup (serveur tué, sous-processus mort).
            # Sans cette trace, l'agent resterait en cache et chaque appel d'outil
            # échouerait sur un ClosedResourceError sans la moindre explication.
            print(f"[agent] session MCP interrompue : {error!r}", file=sys.stderr)


async def build_agent():
    """Construit l'agent à partir des capacités découvertes sur le serveur MCP.

    La session, le sous-processus du serveur et l'agent sont attachés à la boucle
    d'événements sur laquelle cette coroutine tourne. Tous les appels ultérieurs à
    l'agent devront passer par cette même boucle, qui doit donc rester vivante.
    """
    ready = asyncio.get_running_loop().create_future()
    asyncio.ensure_future(_hold_session(ready))
    return await ready
