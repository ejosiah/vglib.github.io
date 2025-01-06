from pathlib import Path
import subprocess
import sys

input_dir = sys.argv[1] if sys.argv[1:] else "../data/shaders"

shader_ext = ['.vert', '.tecs', '.tess' '.frag', '.comp', '.geom']

def compile_shader(path):
    output = f"{path}.spv"
    shader_stage = path.suffix[1:]
    subprocess.run(["glslc", "-g", f"-fshader-stage={shader_stage}", '--target-spv=spv1.6', path, '-o', output])


def visit(path, consume):
    if not path.is_dir():
        if path.suffix in shader_ext:
            consume(path)
    else:
        for p in path.iterdir():
            visit(p, consume)


visit(Path(input_dir), compile_shader)
