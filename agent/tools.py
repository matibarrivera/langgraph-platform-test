from langchain_core.tools import tool


@tool
def obtener_nps_por_cliente(cliente_id: str) -> float:
    """Obtiene el NPS (Net Promoter Score) de un cliente dado su ID."""
    # Aquí iría la lógica para obtener el NPS del cliente desde una base de datos o API.
    # Por simplicidad, retornamos un valor fijo.
    nps_data = {
        "cliente_1": 75.0,
        "cliente_2": 60.0,
        "cliente_3": 85.0,
    }
    return nps_data.get(cliente_id, 0.0)

@tool
def obtener_ids_clientes() -> list[str]:
    """Obtiene una lista de IDs de clientes."""
    # Aquí iría la lógica para obtener los IDs de clientes desde una base de datos o API.
    # Por simplicidad, retornamos una lista fija.
    return ["cliente_1", "cliente_2", "cliente_3"]


@tool
def validar_sucursal(sucursal: str) -> bool:
    """Valida si la sucursal proporcionada es válida."""
    sucursales_validas = ["sucursal_a", "sucursal_b", "sucursal_c"]
    return sucursal in sucursales_validas

tools = [obtener_nps_por_cliente, obtener_ids_clientes, validar_sucursal]

