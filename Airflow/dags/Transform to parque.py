import pandas as pd

# Ruta del archivo Excel
excel_file = "archivo.xlsx"

# Leer el Excel (por defecto la primera hoja)
df = pd.read_excel(excel_file, engine="openpyxl")

# Ruta del archivo Parquet de salida
parquet_file = "archivo.parquet"

# Guardar como Parquet
df.to_parquet(parquet_file, engine="pyarrow", index=False)

print(f"Archivo convertido y guardado como: {parquet_file}")