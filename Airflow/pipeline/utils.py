import re
import pandas as pd

MESES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
}

def extraer_fecha_datetime(texto: str):
    if not isinstance(texto, str): 
        return pd.NaT
    
    # Buscamos el patrón: día + de + mes + de + año
    match = re.search(r"(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})", texto.lower())
    
    if match:
        dia, mes_nom, anio = match.groups()
        mes_num = MESES.get(mes_nom, "01")
        fecha_str = f"{anio}-{mes_num}-{dia.zfill(2)}"
        
        # Hacemos la conversión a datetime aquí mismo
        return pd.to_datetime(fecha_str, format='%Y-%m-%d')
    
    return pd.NaT
