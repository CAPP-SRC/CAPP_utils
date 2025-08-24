#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
CAPP Format Converter - STEP/STL/OBJ → OFF (robusto per CNC-Net)
Dipendenze raccomandate: trimesh, open3d, cadquery (per STEP)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# === Dipendenze opzionali ===
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except Exception:
    TRIMESH_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except Exception:
    CADQUERY_AVAILABLE = False

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("format_converter.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("capp-converter")


class FormatConverter:
    supported_inputs = [".step", ".stp", ".stl", ".obj"]
    output_ext = ".off"

    # --- Utils ---
    @staticmethod
    def _exists(p: Path) -> bool:
        try:
            return p.exists()
        except Exception:
            return False

    def detect_format(self, p: Path) -> Optional[str]:
        ext = p.suffix.lower()
        return ext if ext in self.supported_inputs else None

    def validate_mesh(self, mesh) -> bool:
        """Valida mesh per Open3D o Trimesh."""
        try:
            # Trimesh
            if TRIMESH_AVAILABLE and isinstance(mesh, getattr(trimesh, "Trimesh", ())):
                return mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0
            # Open3D
            if OPEN3D_AVAILABLE and isinstance(mesh, getattr(o3d.geometry, "TriangleMesh", ())):
                return len(mesh.vertices) > 0 and len(mesh.triangles) > 0
        except Exception as e:
            logger.error(f"Errore validazione mesh: {e}")
        return False

    # --- Riparazioni base (Trimesh) ---
    def repair_trimesh(self, mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
        try:
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices()
            mesh.fix_normals()
            mesh.process(validate=True)
        except Exception as e:
            logger.warning(f"Riparazione limitata (procedo comunque): {e}")
        return mesh

    # --- STEP → STL con CadQuery ---
    def step_to_stl(self, step_path: Path, stl_path: Path, tolerance: float = 0.1) -> bool:
        """
        tolerance: ~0.05–0.5 mm a seconda della scala; valori più bassi = più triangoli.
        """
        if not CADQUERY_AVAILABLE:
            logger.error("CadQuery non disponibile: impossibile convertire STEP.")
            return False
        try:
            logger.info(f"STEP → STL: {step_path.name} (tessellation tolerance={tolerance})")
            shape = cq.importers.importStep(str(step_path))
            cq.exporters.export(shape, str(stl_path))
            if not self._exists(stl_path):
                logger.error("STL non generato.")
                return False
            return True
        except Exception as e:
            logger.error(f"Errore STEP→STL: {e}")
            return False

    # --- STL/OBJ → OFF ---
    def mesh_to_off(self, src: Path, dst: Path) -> bool:
        """
        Converte STL/OBJ → OFF.
        Preferisce Trimesh (riparazioni), fallback Open3D.
        """
        # Trimesh path
        if TRIMESH_AVAILABLE:
            try:
                mesh = trimesh.load(str(src), force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    if isinstance(mesh, trimesh.Scene) and len(mesh.geometry):
                        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                    else:
                        logger.error("Impossibile interpretare come mesh triangolare (Trimesh).")
                        raise ValueError("Not a triangular mesh")

                mesh = self.repair_trimesh(mesh)
                if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                    logger.error("Mesh vuota o non triangolare dopo riparazione.")
                    raise ValueError("Empty mesh")

                mesh.export(str(dst), file_type="off")
                if self._exists(dst):
                    return True
                logger.error("Esportazione OFF fallita (Trimesh).")
            except Exception as e:
                logger.warning(f"Trimesh fallita su {src.name}: {e}")

        # Open3D fallback
        if OPEN3D_AVAILABLE:
            try:
                m = o3d.io.read_triangle_mesh(str(src))
                if not self.validate_mesh(m):
                    logger.error("Mesh non valida (Open3D).")
                    return False
                m.remove_duplicated_vertices()
                m.remove_duplicated_triangles()
                m.remove_degenerate_triangles()
                m.remove_non_manifold_edges()
                ok = o3d.io.write_triangle_mesh(str(dst), m, write_ascii=True)
                return bool(ok)
            except Exception as e:
                logger.error(f"Open3D errore su {src.name}: {e}")
                return False

        logger.error("Né Trimesh né Open3D disponibili per convertire mesh.")
        return False

    def convert_file(self, input_path: Path, out_dir: Optional[Path] = None) -> Optional[Path]:
        if not self._exists(input_path):
            logger.error(f"File non trovato: {input_path}")
            return None

        ext = self.detect_format(input_path)
        if not ext:
            logger.error(f"Formato non supportato: {input_path.suffix}")
            return None

        out_dir = out_dir or input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (input_path.stem + self.output_ext)

        logger.info(f"Conversione: {input_path.name} → {out_path.name}")

        if ext in (".step", ".stp"):
            temp_stl = out_dir / f"{input_path.stem}__tmp.stl"
            try:
                if not self.step_to_stl(input_path, temp_stl, tolerance=0.1):
                    return None
                if not self.mesh_to_off(temp_stl, out_path):
                    return None
                return out_path
            finally:
                if self._exists(temp_stl):
                    try:
                        temp_stl.unlink()
                    except Exception:
                        pass

        if ext in (".stl", ".obj"):
            return out_path if self.mesh_to_off(input_path, out_path) else None

        return None

    def batch_convert(self, input_dir: Path, out_dir: Optional[Path] = None) -> List[Path]:
        if not self._exists(input_dir) or not input_dir.is_dir():
            logger.error(f"Directory non valida: {input_dir}")
            return []

        out_dir = out_dir or input_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        converted: List[Path] = []
        for ext in self.supported_inputs:
            for f in input_dir.glob(f"*{ext}"):
                logger.info(f"Trovato: {f.name}")
                r = self.convert_file(f, out_dir)
                if r is not None:
                    converted.append(r)

        return converted

    def check_dependencies(self) -> dict:
        return {
            "trimesh": TRIMESH_AVAILABLE,
            "open3d": OPEN3D_AVAILABLE,
            "cadquery": CADQUERY_AVAILABLE,
        }


def main():
    p = argparse.ArgumentParser(
        description="Convertitore automatico formati 3D per CNC-Net (.step/.stl/.obj → .off)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python format_converter.py modello.step
  python format_converter.py modello.stl -o out_dir/
  python format_converter.py --batch ./modelli
  python format_converter.py --check
""",
    )
    p.add_argument("input", nargs="?", help="File o directory di input")
    p.add_argument("-o", "--output", help="Directory di output")
    p.add_argument("-b", "--batch", action="store_true", help="Conversione batch della directory")
    p.add_argument("-c", "--check", action="store_true", help="Verifica dipendenze")
    p.add_argument("-v", "--verbose", action="store_true", help="Log verboso")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    conv = FormatConverter()

    if args.check:
        deps = conv.check_dependencies()
        print("\n🔍 Dipendenze:")
        for k, ok in deps.items():
            print(f"  {k:9s}: {'✅' if ok else '❌'}")
        if not any(deps.values()):
            print("\nInstalla almeno una libreria utile:\n  pip install trimesh open3d cadquery")
        return

    if not args.input:
        p.print_help()
        sys.exit(0)

    in_path = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    if args.batch:
        print(f"🔄 Batch in: {in_path}")
        results = conv.batch_convert(in_path, out_dir)
        if results:
            print("\n✅ Completato. Generati:")
            for r in results:
                print(f"  • {r}")
        else:
            print("❌ Nessun file convertito")
        return

    # singolo file
    print(f"🔄 Singolo file: {in_path}")
    res = conv.convert_file(in_path, out_dir)
    if res:
        print(f"✅ OK: {res}")
    else:
        print("❌ Conversione fallita")
        sys.exit(1)


if __name__ == "__main__":
    main()
