import pandas as pd
import numpy as np
import os
import logging

log = logging.getLogger(__name__)

# --- CONSTANTES DE ESTANDARIZACIÓN ---
# Usaremos nombres fijos para evitar que el merge falle por nombres distintos
KEY_COL = 'KEY_ID' 

def enrich_master(path_maestra, path_rescatados, output_path):
    """
    Lee fuentes (Excel/Parquet), inyecta nombres rescatados y genera un Parquet robusto.
    """
    log.info(f"🚀 Iniciando Enriquecimiento de Maestra: {os.path.basename(path_maestra)}")

    def read_flexible(path):
        if not os.path.exists(path):
            log.warning(f"⚠️ Archivo no encontrado: {path}. Creando DF vacío.")
            return pd.DataFrame()
        return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_excel(path)

    df_m = read_flexible(path_maestra)
    df_n = read_flexible(path_rescatados)

    # 1. Normalización de columnas (Todo a Mayúsculas y sin espacios)
    for df in [df_m, df_n]:
        if not df.empty:
            df.columns = [str(c).strip().upper() for c in df.columns]

    # 2. Identificar la columna 'DS' (El DNI de los registros)
    # Buscamos 'DS' o 'DS_INSTALACION' o cualquier cosa que contenga 'DS'
    for df in [df_m, df_n]:
        if not df.empty:
            ds_col = next((c for c in df.columns if 'DS' in c), None)
            if ds_col:
                df['DS_JOIN'] = df[ds_col].astype(str).str.strip().str.upper()
                df.drop_duplicates(subset=['DS_JOIN'], keep='last', inplace=True)

    # 3. Merge de nombres rescatados
    if not df_n.empty and 'DS_JOIN' in df_m.columns and 'DS_JOIN' in df_n.columns:
        # Buscamos la columna de nombres en los rescatados
        col_nombre_n = next((c for c in df_n.columns if 'NODE' in c or 'NAME' in c), None)
        
        if col_nombre_n:
            log.info(f"Inyectando nombres desde columna: {col_nombre_n}")
            df_final = pd.merge(df_m, df_n[['DS_JOIN', col_nombre_n]], on='DS_JOIN', how='left')
            
            # Lógica de reemplazo (Visto en tu script de Enriquecimiento V2)
            if 'NODE NAME' in df_final.columns:
                df_final['NODE NAME'] = df_final[col_nombre_n].fillna(df_final['NODE NAME'])
            else:
                df_final['NODE NAME'] = df_final[col_nombre_n]
            
            df_final.drop(columns=[col_nombre_n, 'DS_JOIN'], errors='ignore', inplace=True)
        else:
            df_final = df_m
    else:
        df_final = df_m

    # 4. Limpieza final y blindaje Parquet
    for col in df_final.columns:
        df_final[col] = df_final[col].astype(str).replace(['nan', 'None', '<NA>', 'NAT'], '')

    final_output = os.path.splitext(output_path)[0] + ".parquet"
    df_final.to_parquet(final_output, engine='pyarrow', index=False, compression='snappy')
    
    log.info(f"✨ Maestra V2 generada: {final_output} ({len(df_final)} registros)")
    return final_output


