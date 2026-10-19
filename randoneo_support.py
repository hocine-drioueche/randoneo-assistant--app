"""Serveur MCP du support Randoneo, réduit à deux capacités.

Deux tools : consulter une commande, et ouvrir un ticket — la seule action qui
écrit quelque chose, donc la seule soumise à validation humaine côté agent.
Les données sont en dur : ce lab porte sur le branchement, pas sur le métier.

Lancement direct (stdio) : python randoneo_support.py
"""
from fastmcp import FastMCP

mcp = FastMCP("randoneo-support")

ORDERS = {
    "RND-10234": {"status": "expédiée", "shipping_method": "standard",
                  "tracking_number": "TRK-FR-884512", "eta": "2026-06-15"},
    "RND-10235": {"status": "en préparation", "shipping_method": "express",
                  "tracking_number": None, "eta": "2026-06-13"},
    "RND-10236": {"status": "livrée", "shipping_method": "standard",
                  "tracking_number": "TRK-FR-871203", "eta": "2026-06-05"},
}

TICKETS = []


@mcp.tool
def get_order_status(order_id: str) -> dict:
    """Donne l'état d'une commande Randoneo à partir de son identifiant.

    order_id : identifiant de commande, par ex. « RND-10234 ».
    À utiliser dès qu'un client demande où en est sa commande, ou parle d'un colis.
    """
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"erreur": f"Commande « {order_id} » introuvable. Vérifiez le format RND-XXXXX."}
    return {"order_id": order_id.strip().upper(), **order}


@mcp.tool
def create_ticket(intent: str, summary: str, order_id: str | None = None) -> str:
    """Ouvre un ticket au support Randoneo pour escalader une demande vers un humain.

    intent : nature de la demande (return_refund, payment_issue, order_tracking…).
    summary : résumé de la demande du client, en une phrase.
    order_id : identifiant de la commande concernée, si le client en mentionne une.
    Action sensible : l'agent la soumet à validation avant de l'exécuter.
    """
    TICKETS.append({"intent": intent, "summary": summary, "order_id": order_id})
    return f"Ticket TICK-{len(TICKETS):03d} ouvert ({intent})."


if __name__ == "__main__":
    mcp.run()  # stdio : l'hôte lance ce fichier en sous-processus
