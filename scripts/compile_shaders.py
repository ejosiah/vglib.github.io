from pathlib import Path
import subprocess
import sys

input_dir = sys.argv[1] if sys.argv[1:] else "../data/shaders"

shader_ext = ['.vert', '.tesc', '.tese', '.frag', '.comp', '.geom']
failures = []

def compile_shader(path):
    output = f"{path}.spv"
    shader_stage = path.suffix[1:]
    command = ["glslc", "-g", f"-fshader-stage={shader_stage}", "--target-spv=spv1.6", str(path), "-o", output]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        failures.append(path)
        print("FAILED:", " ".join(command))
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


def visit(path, consume):
    if not path.is_dir():
        if path.suffix in shader_ext:
            consume(path)
    else:
        for p in path.iterdir():
            visit(p, consume)


visit(Path(input_dir), compile_shader)

if failures:
    sys.exit(1)
