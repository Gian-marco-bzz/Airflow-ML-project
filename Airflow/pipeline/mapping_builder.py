import pandas as pd
import os
import glob

def build_identifier_master(input_folder, output_path):
    archivos = glob.glob(os.path.join(input_folder, '*.xlsx'))
    list_dfs = []

    if not archivos:
        print(f"⚠️ No se encontraron archivos Excel en {input_folder}")
        return None

    for archivo in archivos:
        print(f"📖 Procesando: {os.path.basename(archivo)}")
        # 1. Detectar cabecera de forma eficiente
        df_preview = pd.read_excel(archivo, nrows=10, header=None)
        header_row = 0
        
        for i, row in df_preview.iterrows():
            row_str = ' '.join(str(v) for v in row.values).upper()
            if 'NODE NAME' in row_str or 'DS_INSTALA' in row_str:
                header_row = i
                break

        # 2. Cargar datos
        df = pd.read_excel(archivo, skiprows=header_row)
        df.columns = [str(c).strip().upper() for c in df.columns]

        # 3. Identificar columnas dinámicamente
        col_nodo = next((c for c in df.columns if 'NODE' in c or 'SERVICIO' in c), None)
        col_ds = next((c for c in df.columns if 'DS' in c and 'INSTALA' in c), None)

        if col_nodo and col_ds:
            subset = df[[col_nodo, col_ds]].dropna().copy()
            subset.columns = ['NODE NAME', 'DS_INSTALACION']
            
            # Normalización vectorizada
            subset['NODE NAME'] = subset['NODE NAME'].astype(str).str.strip().str.upper()
            subset['DS_INSTALACION'] = subset['DS_INSTALACION'].astype(str).str.strip().str.upper()
            
            subset = subset[subset['DS_INSTALACION'] != ""]
            list_dfs.append(subset)

    if not list_dfs:
        print("❌ No se encontraron datos válidos para procesar.")
        return None

    # 4. Concatenar y eliminar duplicados
    df_out = pd.concat(list_dfs, ignore_index=True)
    df_out = df_out.drop_duplicates(subset=['NODE NAME'], keep='last')

    # 5. Blindaje de tipos para Parquet (Aseguramos que todo sea string)
    for col in df_out.columns:
        df_out[col] = df_out[col].astype(str).replace(['nan', 'None', '<NA>'], '')

    # ---------------------------------------------------------
    # 6. FORZAR EXTENSIÓN .PARQUET REAL
    # ---------------------------------------------------------
    base_sin_ext = os.path.splitext(output_path)[0]
    final_output = f"{base_sin_ext}.parquet"
    
    # IMPORTANTE: Cambiado df_resultado por df_out
    df_out.to_parquet(final_output, engine='pyarrow', index=False, compression='snappy')
    
    print(f"✅ Mapping guardado exitosamente como: {final_output}")
    print(f"📊 Registros únicos mapeados: {len(df_out)}")
    
    return final_output