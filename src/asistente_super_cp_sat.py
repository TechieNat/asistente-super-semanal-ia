"""Asistente inteligente para la planeación del súper semanal.

Lee seis archivos CSV, entrena un árbol de decisión y resuelve la selección
semanal de recetas y paquetes mediante Google OR-Tools CP-SAT.

Uso:
  python asistente_super_cp_sat.py --datos . --hogar H001
  python asistente_super_cp_sat.py --datos . --hogar H004 --presupuesto 50

Instalación:
  pip install -r requirements.txt
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from ortools.sat.python import cp_model

ARCHIVOS = {
    "hogares": "hogares.csv",
    "ingredientes": "ingredientes.csv",
    "recetas": "recetas.csv",
    "receta_ingredientes": "receta_ingredientes.csv",
    "catalogo": "catalogo_supermercado.csv",
    "historial": "historial_consumo.csv",
}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def cargar_datos(carpeta: Path) -> dict[str, pd.DataFrame]:
    datos = {}
    faltantes = []
    for nombre, archivo in ARCHIVOS.items():
        ruta = carpeta / archivo
        if not ruta.exists():
            faltantes.append(archivo)
        else:
            datos[nombre] = pd.read_csv(ruta, encoding="utf-8-sig")
    if faltantes:
        raise FileNotFoundError("Faltan archivos: " + ", ".join(faltantes))
    return datos


def validar_datos(d: dict[str, pd.DataFrame]) -> None:
    hogar_ids = set(d["hogares"].hogar_id)
    ing_ids = set(d["ingredientes"].ingrediente_id)
    receta_ids = set(d["recetas"].receta_id)
    assert set(d["receta_ingredientes"].receta_id) <= receta_ids
    assert set(d["receta_ingredientes"].ingrediente_id) <= ing_ids
    assert set(d["catalogo"].ingrediente_id) <= ing_ids
    assert set(d["historial"].hogar_id) <= hogar_ids
    assert set(d["historial"].ingrediente_id) <= ing_ids
    if (d["catalogo"].precio_mxn < 0).any():
        raise ValueError("El catálogo contiene precios negativos.")


def entrenar_modelo_consumo(d: dict[str, pd.DataFrame]):
    base = d["historial"].merge(
        d["hogares"][["hogar_id", "numero_integrantes", "adultos", "ninos"]],
        on="hogar_id", how="left"
    )
    features_num = ["numero_integrantes", "adultos", "ninos", "cantidad_comprada",
                    "inventario_final", "aceptacion_1_5"]
    features_cat = ["ingrediente_id"]
    X = base[features_num + features_cat]
    y = base["cantidad_consumida"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    prep = ColumnTransformer([
        ("num", "passthrough", features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ])
    modelo = Pipeline([
        ("preparacion", prep),
        ("arbol", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)),
    ])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    metricas = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }
    return modelo, metricas


def predecir_consumo(modelo, d: dict[str, pd.DataFrame], hogar_id: str) -> pd.DataFrame:
    h = d["hogares"].set_index("hogar_id").loc[hogar_id]
    hist = d["historial"][d["historial"].hogar_id == hogar_id]
    ult = (hist.sort_values("semana_inicio").groupby("ingrediente_id", as_index=False).tail(1))
    promedio = hist.groupby("ingrediente_id", as_index=False).agg(
        cantidad_comprada=("cantidad_comprada", "mean"),
        inventario_final=("inventario_final", "mean"),
        aceptacion_1_5=("aceptacion_1_5", "mean"),
    )
    marco = d["ingredientes"][["ingrediente_id", "unidad"]].merge(promedio, on="ingrediente_id", how="left")
    inv_ult = ult[["ingrediente_id", "inventario_final"]].rename(columns={"inventario_final":"inventario_actual"})
    marco = marco.merge(inv_ult, on="ingrediente_id", how="left").fillna(0)
    marco["numero_integrantes"] = int(h.numero_integrantes)
    marco["adultos"] = int(h.adultos)
    marco["ninos"] = int(h.ninos)
    cols = ["numero_integrantes", "adultos", "ninos", "cantidad_comprada",
            "inventario_final", "aceptacion_1_5", "ingrediente_id"]
    marco["consumo_predicho"] = modelo.predict(marco[cols]).clip(min=0)
    return marco


def separar(texto) -> set[str]:
    if pd.isna(texto) or str(texto).strip().lower() in {"", "ninguna", "ninguno"}:
        return set()
    return {x.strip().lower() for x in str(texto).split(";") if x.strip()}


def recetas_permitidas(d, hogar_id: str) -> tuple[list[str], list[str]]:
    h = d["hogares"].set_index("hogar_id").loc[hogar_id]
    preferencia = str(h.preferencia_alimentaria).lower()
    permitidos = {
        "vegana": {"vegana"},
        "vegetariana": {"vegana", "vegetariana"},
        "pescetariana": {"vegana", "vegetariana", "pescetariana"},
        "omnívora": {"vegana", "vegetariana", "pescetariana", "omnívora"},
        "omnivora": {"vegana", "vegetariana", "pescetariana", "omnívora"},
    }.get(preferencia, {preferencia})
    candidatos = set(d["recetas"].loc[
        d["recetas"].tipo_alimentacion.str.lower().isin(permitidos), "receta_id"
    ])
    alergias = separar(h.alergenos_excluidos)
    if alergias:
        mapa_alerg = d["ingredientes"].set_index("ingrediente_id").alergeno_asociado.to_dict()
        prohibidas = set()
        for row in d["receta_ingredientes"].itertuples():
            alergenos_ing = separar(mapa_alerg[row.ingrediente_id])
            if alergias & alergenos_ing:
                prohibidas.add(row.receta_id)
        candidatos -= prohibidas
    excluidas = sorted(set(d["recetas"].receta_id) - candidatos)
    return sorted(candidatos), excluidas


def preparar_productos(d):
    disponibles = d["catalogo"][d["catalogo"].disponible == 1].copy()
    disponibles["precio_centavos"] = (disponibles.precio_mxn * 100).round().astype(int)
    # Para mantener el prototipo legible, usamos una alternativa disponible por ingrediente:
    # la de menor costo por unidad. CP-SAT decide cuántos paquetes comprar.
    disponibles["costo_unidad"] = disponibles.precio_mxn / disponibles.cantidad_por_paquete
    elegidos = (disponibles.sort_values(["ingrediente_id", "costo_unidad", "precio_mxn"])
                .groupby("ingrediente_id", as_index=False).first())
    return elegidos.set_index("ingrediente_id", drop=False)


def construir_y_resolver(d, hogar_id, predicciones, presupuesto_mxn=None, incluir_presupuesto=True):
    hogar = d["hogares"].set_index("hogar_id").loc[hogar_id]
    presupuesto = float(presupuesto_mxn if presupuesto_mxn is not None else hogar.presupuesto_semanal_mxn)
    recetas_ok, recetas_excluidas = recetas_permitidas(d, hogar_id)
    productos = preparar_productos(d)
    recetas = d["recetas"].set_index("receta_id")
    ri = d["receta_ingredientes"]
    inventario = predicciones.set_index("ingrediente_id").inventario_actual.to_dict()

    modelo = cp_model.CpModel()
    # Una receta diferente por día. Cada receta se usa como máximo una vez para favorecer variedad.
    x = {(r, dia): modelo.new_bool_var(f"x_{r}_{dia}") for r in recetas_ok for dia in range(7)}
    for dia in range(7):
        modelo.add(sum(x[r, dia] for r in recetas_ok) == 1)
    for r in recetas_ok:
        modelo.add(sum(x[r, dia] for dia in range(7)) <= 1)

    # Demanda en la unidad de cada ingrediente. Los CSV ya usan cantidades enteras.
    requerimiento = {}
    compra = {}
    sobrante = {}
    ingredientes_usados = sorted(set(ri[ri.receta_id.isin(recetas_ok)].ingrediente_id))
    for ing in ingredientes_usados:
        filas = ri[ri.ingrediente_id == ing]
        expr = []
        for f in filas.itertuples():
            if f.receta_id in recetas_ok:
                for dia in range(7):
                    expr.append(int(round(f.cantidad_requerida)) * x[f.receta_id, dia])
        requerimiento[ing] = sum(expr)
        inv = int(round(inventario.get(ing, 0)))
        if ing in productos.index:
            paquete = int(round(productos.loc[ing, "cantidad_por_paquete"]))
            compra[ing] = modelo.new_int_var(0, 50, f"paquetes_{ing}")
            sobrante[ing] = modelo.new_int_var(0, 100000, f"sobrante_{ing}")
            modelo.add(inv + paquete * compra[ing] - requerimiento[ing] == sobrante[ing])
        else:
            # Si un ingrediente no tiene producto disponible, solo puede cubrirse con inventario.
            modelo.add(requerimiento[ing] <= inv)

    costo = sum(int(productos.loc[i, "precio_centavos"]) * compra[i] for i in compra)
    if incluir_presupuesto:
        modelo.add(costo <= int(round(presupuesto * 100)))

    # Pesos transparentes. Costo en centavos + sobrante + baja aceptación.
    penal_baja_aceptacion = sum(
        (5 - int(recetas.loc[r, "nivel_aceptacion_base_1_5"])) * 500 * x[r, dia]
        for r in recetas_ok for dia in range(7)
    )
    penal_sobrante = sum(sobrante.values())
    modelo.minimize(costo + penal_sobrante + penal_baja_aceptacion)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_search_workers = 8
    estado = solver.solve(modelo)
    return {
        "estado": estado, "solver": solver, "modelo": modelo, "x": x,
        "compra": compra, "sobrante": sobrante, "costo": costo,
        "recetas_ok": recetas_ok, "recetas_excluidas": recetas_excluidas,
        "productos": productos, "presupuesto": presupuesto,
    }


def diagnosticar_infactibilidad(d, hogar_id, predicciones, presupuesto):
    candidatos, excluidas = recetas_permitidas(d, hogar_id)
    causas = []
    if len(candidatos) < 7:
        causas.append(f"Solo hay {len(candidatos)} recetas distintas compatibles y se requieren 7.")
    relajado = construir_y_resolver(d, hogar_id, predicciones, presupuesto, incluir_presupuesto=False)
    if relajado["estado"] in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        minimo = relajado["solver"].value(relajado["costo"]) / 100
        if minimo > presupuesto:
            causas.append(f"El presupuesto es insuficiente. Mínimo estimado: ${minimo:,.2f} MXN.")
    else:
        causas.append("Aun sin límite de presupuesto no existe solución: revise disponibilidad, alergias y variedad.")
    if not causas:
        causas.append("La combinación de restricciones no tiene solución con los datos actuales.")
    return causas


def exportar_resultados(d, hogar_id, resultado, carpeta_salida: Path):
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    solver = resultado["solver"]
    recetas = d["recetas"].set_index("receta_id")
    menu = []
    for (r, dia), var in resultado["x"].items():
        if solver.value(var):
            menu.append({"dia": DIAS[dia], "receta_id": r, "receta": recetas.loc[r, "nombre"]})
    menu = pd.DataFrame(menu).sort_values("dia", key=lambda s: s.map({d:i for i,d in enumerate(DIAS)}))
    compras = []
    for ing, var in resultado["compra"].items():
        n = solver.value(var)
        if n:
            p = resultado["productos"].loc[ing]
            compras.append({
                "ingrediente_id": ing, "producto_id": p.producto_id, "marca": p.marca,
                "paquetes": n, "cantidad_por_paquete": p.cantidad_por_paquete,
                "precio_unitario_mxn": p.precio_mxn,
                "subtotal_mxn": round(n * p.precio_mxn, 2),
                "sobrante_estimado": solver.value(resultado["sobrante"][ing]),
            })
    compras = pd.DataFrame(compras)
    menu.to_csv(carpeta_salida / f"menu_{hogar_id}.csv", index=False, encoding="utf-8-sig")
    compras.to_csv(carpeta_salida / f"compras_{hogar_id}.csv", index=False, encoding="utf-8-sig")
    return menu, compras


def main():
    parser = argparse.ArgumentParser(description="Planeación semanal con predicción y CP-SAT")
    parser.add_argument("--datos", type=Path, default=Path("."), help="Carpeta con los seis CSV")
    parser.add_argument("--hogar", default="H001", help="Identificador del hogar")
    parser.add_argument("--presupuesto", type=float, default=None, help="Sobrescribe presupuesto para probar infactibilidad")
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    args = parser.parse_args()

    d = cargar_datos(args.datos)
    validar_datos(d)
    if args.hogar not in set(d["hogares"].hogar_id):
        raise ValueError(f"Hogar inexistente: {args.hogar}")

    modelo_ml, metricas = entrenar_modelo_consumo(d)
    predicciones = predecir_consumo(modelo_ml, d, args.hogar)
    predicciones.to_csv(args.salida / f"predicciones_{args.hogar}.csv", index=False, encoding="utf-8-sig") if args.salida.exists() else None

    resultado = construir_y_resolver(d, args.hogar, predicciones, args.presupuesto)
    nombres_estado = {
        cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    estado = nombres_estado.get(resultado["estado"], str(resultado["estado"]))
    print("\n=== MODELO PREDICTIVO ===")
    print(" | ".join(f"{k}: {v:.3f}" for k, v in metricas.items()))
    print(f"\n=== OPTIMIZACIÓN CP-SAT: {estado} ===")

    if resultado["estado"] in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        args.salida.mkdir(parents=True, exist_ok=True)
        predicciones.to_csv(args.salida / f"predicciones_{args.hogar}.csv", index=False, encoding="utf-8-sig")
        menu, compras = exportar_resultados(d, args.hogar, resultado, args.salida)
        costo = resultado["solver"].value(resultado["costo"]) / 100
        print(menu.to_string(index=False))
        print(f"\nCosto total: ${costo:,.2f} MXN / Presupuesto: ${resultado['presupuesto']:,.2f} MXN")
        print(f"Archivos creados en: {args.salida.resolve()}")
    elif resultado["estado"] == cp_model.INFEASIBLE:
        print("No se genera una recomendación incompleta.")
        print("Diagnóstico:")
        for causa in diagnosticar_infactibilidad(d, args.hogar, predicciones, resultado["presupuesto"]):
            print(" -", causa)
        print("Protección: las alergias no se relajan automáticamente.")
    else:
        print("No se obtuvo una solución concluyente. Revise el modelo o aumente el tiempo de solución.")


if __name__ == "__main__":
    main()
