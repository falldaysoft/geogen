"""Shader compilation and management utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from OpenGL import GL


@dataclass
class ShaderProgram:
    """Compiled shader program with uniform management."""

    program_id: int
    _uniform_cache: dict[str, int] = field(default_factory=dict, repr=False)

    def use(self) -> None:
        """Activate this shader program."""
        GL.glUseProgram(self.program_id)

    def get_uniform_location(self, name: str) -> int:
        """Get uniform location, caching for performance."""
        if name not in self._uniform_cache:
            loc = GL.glGetUniformLocation(self.program_id, name)
            self._uniform_cache[name] = loc
        return self._uniform_cache[name]

    def set_uniform(self, name: str, value: Any) -> None:
        """Set a uniform value by name."""
        loc = self.get_uniform_location(name)
        if loc == -1:
            return  # Uniform not found or optimized out

        if isinstance(value, bool):
            GL.glUniform1i(loc, int(value))
        elif isinstance(value, int):
            GL.glUniform1i(loc, value)
        elif isinstance(value, float):
            GL.glUniform1f(loc, value)
        elif isinstance(value, (tuple, list)):
            if len(value) == 2:
                GL.glUniform2f(loc, *value)
            elif len(value) == 3:
                GL.glUniform3f(loc, *value)
            elif len(value) == 4:
                GL.glUniform4f(loc, *value)
        elif isinstance(value, np.ndarray):
            if value.shape == (3,):
                GL.glUniform3fv(loc, 1, value.astype(np.float32))
            elif value.shape == (4,):
                GL.glUniform4fv(loc, 1, value.astype(np.float32))
            elif value.shape == (3, 3):
                # Transpose because numpy is row-major, OpenGL expects column-major
                GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, value.astype(np.float32))
            elif value.shape == (4, 4):
                # Transpose because numpy is row-major, OpenGL expects column-major
                GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, value.astype(np.float32))

    def set_uniform_array(self, name: str, values: list, component_size: int = 3) -> None:
        """Set an array uniform (e.g., uLightPositions[0], [1], etc.)."""
        for i, value in enumerate(values):
            self.set_uniform(f"{name}[{i}]", value)

    def delete(self) -> None:
        """Delete the shader program."""
        if self.program_id:
            GL.glDeleteProgram(self.program_id)
            self.program_id = 0


class ShaderCompiler:
    """Compiles and caches shader programs."""

    _shader_dir: Path = Path(__file__).parent
    _cache: dict[str, ShaderProgram] = {}

    @classmethod
    def compile(
        cls,
        vertex_source: str,
        fragment_source: str,
        name: str = "unnamed",
    ) -> ShaderProgram:
        """Compile vertex and fragment shaders into a program.

        Args:
            vertex_source: GLSL vertex shader source code
            fragment_source: GLSL fragment shader source code
            name: Optional name for error messages

        Returns:
            Compiled ShaderProgram

        Raises:
            RuntimeError: If compilation or linking fails
        """
        # Compile vertex shader
        vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vertex_shader, vertex_source)
        GL.glCompileShader(vertex_shader)

        if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(vertex_shader).decode()
            GL.glDeleteShader(vertex_shader)
            raise RuntimeError(f"Vertex shader compilation failed ({name}):\n{error}")

        # Compile fragment shader
        fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fragment_shader, fragment_source)
        GL.glCompileShader(fragment_shader)

        if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(fragment_shader).decode()
            GL.glDeleteShader(vertex_shader)
            GL.glDeleteShader(fragment_shader)
            raise RuntimeError(f"Fragment shader compilation failed ({name}):\n{error}")

        # Link program
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex_shader)
        GL.glAttachShader(program, fragment_shader)
        GL.glLinkProgram(program)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            error = GL.glGetProgramInfoLog(program).decode()
            GL.glDeleteShader(vertex_shader)
            GL.glDeleteShader(fragment_shader)
            GL.glDeleteProgram(program)
            raise RuntimeError(f"Shader program linking failed ({name}):\n{error}")

        # Clean up individual shaders (they're now part of the program)
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

        return ShaderProgram(program_id=program)

    @classmethod
    def load(cls, vertex_file: str, fragment_file: str) -> ShaderProgram:
        """Load and compile shaders from files.

        Args:
            vertex_file: Filename of vertex shader (relative to shaders dir)
            fragment_file: Filename of fragment shader (relative to shaders dir)

        Returns:
            Compiled ShaderProgram
        """
        cache_key = f"{vertex_file}:{fragment_file}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        vertex_path = cls._shader_dir / vertex_file
        fragment_path = cls._shader_dir / fragment_file

        vertex_source = vertex_path.read_text()
        fragment_source = fragment_path.read_text()

        program = cls.compile(vertex_source, fragment_source, name=cache_key)
        cls._cache[cache_key] = program
        return program

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the shader cache and delete all cached programs."""
        for program in cls._cache.values():
            program.delete()
        cls._cache.clear()
